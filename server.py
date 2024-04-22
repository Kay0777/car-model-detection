from fastapi.responses import JSONResponse
from uvicorn import Config, Server
from pydantic import BaseModel
from fastapi import FastAPI

from uuid import uuid4
import redis
import json

from config import CONF

# _______________________________________________________
IMAGE_BATCH_NAME: str = CONF['IMAGE_BATCH_NAME']
IMAGE_HANDLE_SIZE: int = CONF['IMAGE_HANDLE_SIZE']
# _______________________________________________________


class Image(BaseModel):
    image: str


app: FastAPI = FastAPI()
Redis_Client = redis.StrictRedis(
    host=CONF['REDIS_HOST'],
    port=CONF['REDIS_PORT'],
    db=CONF['REDIS_DB']
)


@app.post(path='/')
def add_Image(image: Image) -> JSONResponse:
    batchSize: int = Redis_Client.llen(IMAGE_BATCH_NAME)
    if batchSize == IMAGE_HANDLE_SIZE:
        return JSONResponse(
            status_code=202,
            content={
                "message": "Tasks batch size is full!",
                "id": '',
                "result": {},
            }
        )

    imageUUID: str = uuid4().hex
    convertedData = json.dumps((imageUUID, image.image))
    Redis_Client.rpush(IMAGE_BATCH_NAME, convertedData)

    # Add the image and its identifier to the queue
    return JSONResponse(
        content={
            "message": "Image received and queued for processing!",
            "id": imageUUID,
            "result": {},
        }, status_code=201
    )


@app.get('/{uuid}')
async def get_Car_Model(uuid: str) -> JSONResponse:
    result = Redis_Client.get(name=uuid)
    if result is None:
        return JSONResponse(
            status_code=202,
            content={
                "message": "Result not available yet!",
                "id": uuid,
                "result": {},
            }
        )
    return JSONResponse(
        status_code=200,
        content={
            "message": "Successfully done!",
            "id": uuid,
            "result": json.loads(result),
        }
    )


if __name__ == "__main__":
    config = Config(app='server:app',
                    host=CONF['HOST'],
                    port=CONF['PORT'],
                    workers=CONF['WORKERS'])
    server = Server(config)
    server.run()
