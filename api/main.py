
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, Response
from threading import Thread
from config import CONF
import logging
import multiprocessing as mp

from tasks import (
    runAnalysis,
    PreProcessing,
    ProcessedImageResult,
)


def API_WITH_Flask():
    from flask import Flask, Response
    from flask import request, jsonify

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app = Flask(__name__)
    app.logger.setLevel(logging.WARNING)

    @app.route('/', methods=['POST'])
    def predict_car() -> Response:
        uuid, isBusy = PreProcessing(
            base64_encoded_image=request.json['image'])
        if isBusy:
            return jsonify({
                "message": "Tasks batch size is full!",
                "id": '',
                "result": {},
            }), 202

        # Add the image and its identifier to the queue
        return jsonify({
            "message": "Image received and queued for processing!",
            "id": uuid,
            "result": {},
        }), 201

    @app.route('/<uuid>', methods=['GET'])
    def get_car_model(uuid: str) -> Response:
        result = ProcessedImageResult(imageUUID=uuid)
        if result is None:
            logging.warning(f"Result not available for ID: {uuid}")
            return jsonify({
                "message": "Result not available yet!",
                "id": uuid,
                "result": {},
            }), 202

        logging.info(f"Result found for ID: {uuid}")
        return jsonify({
            "message": "Successfully done!",
            "id": uuid,
            "result": result,
        }), 200

    # Start analyze tasks [images]
    # ____________________________________________________________
    thread = Thread(target=runAnalysis, daemon=True)
    thread.start()
    # ____________________________________________________________


app = FastAPI()

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.on_event("startup")
def startup_event():
    # Start the background thread
    thread = Thread(target=runAnalysis)
    thread.daemon = True
    thread.start()

# Define a Pydantic model for your data


class ImageData(BaseModel):
    image: str  # assuming the image is sent as a base64 encoded string


@app.post('/')
async def predict_car(image_data: ImageData) -> Response:
    uuid, isBusy = PreProcessing(base64_encoded_image=image_data.image)
    if isBusy:
        return JSONResponse(
            content={
                "message": "Tasks batch size is full!",
                "id": '',
                "result": {},
            }, status_code=202
        )
    # Add the image and its identifier to the queue
    return JSONResponse(
        content={
            "message": "Image received and queued for processing!",
            "id": uuid,
            "result": {},
        }, status_code=201
    )


@app.get('/{uuid}')
async def get_car_model(uuid: str) -> Response:
    result = ProcessedImageResult(imageUUID=uuid)
    if result is None:
        logging.warning(f"Result not available for ID: {uuid}")
        return JSONResponse(
            content={
                "message": "Result not available yet!",
                "id": uuid,
                "result": {},
            }, status_code=202
        )
    logging.info(f"Result found for ID: {uuid}")
    return JSONResponse(
        content={
            "message": "Successfully done!",
            "id": uuid,
            "result": result,
        }, status_code=200
    )


# from gunicorn.app.base import BaseApplication
# class StandaloneGunicornApplication(BaseApplication):
#     def __init__(self, app, options=None):
#         self.application = app
#         self.options = options or {}
#         super(StandaloneGunicornApplication, self).__init__()

#     def load_config(self):
#         config = {key: value for key, value in self.options.items()
#                   if key in self.cfg.settings and value is not None}
#         for key, value in config.items():
#             self.cfg.set(key.lower(), value)

#     def load(self):
#         return self.application

def main():
    # Start analyze tasks [images]
    # ____________________________________________________________
    uvicorn.run(
        app='main:app',
        host=CONF['HOST'],
        port=CONF['PORT'],
        reload=False,
        workers=8
    )


if __name__ == '__main__':
    main()
    # API_WITH_Flask()
    # # ____________________________________________________________
    # # Get the port number from command line argument
    # app.run(
    #     debug=False,
    #     host=CONF['HOST'],
    #     port=CONF['PORT']
    # )
    # # ____________________________________________________________

    # thread = Thread(target=runAnalysis, daemon=True)
    # thread.start()
    # # ____________________________________________________________
    # options = {
    #     'bind': '0.0.0.0:5000',
    #     'workers'
    #     'threads': 8,
    #     'worker_class': 'uvicorn.workers.UvicornWorker',
    # }
    # StandaloneGunicornApplication(app, options).run()
