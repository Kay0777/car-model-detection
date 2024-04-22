from torchvision import transforms
from openvino.runtime import Core
from base64 import b64decode
from io import BytesIO
from PIL import Image
import numpy as np
import redis
import time
import json

from utils import (
    Read_Classes_With_ID_From_Json_File,
    Read_Classes_From_Classification_File,
)

from config import CONF

# _______________________________________________________
MODEL_XML_PATH: str = CONF['OPENVINO_MODEL_XML']
MODEL_DEVICE: str = CONF['OPENVINO_MODEL_DEVICE']

CLASS_NAMES: list[str] = Read_Classes_From_Classification_File(
    filename=CONF['CLASS_NAMES'])
CLASS_NAMES_WITH_IDS: dict[str, int] = Read_Classes_With_ID_From_Json_File(
    filename=CONF['CLASS_NAMES_WITH_IDS'])

IMAGE_WAIT_TTL: int = CONF['IMAGE_WAIT_TTL']
IMAGE_BATCH_NAME: str = CONF['IMAGE_BATCH_NAME']

IMAGE_SIMILARITY_K: float = CONF['IMAGE_SIMILARITY_K']
IMAGE_BATCH_SIZE: int = CONF['IMAGE_BATCH_SIZE']
IMAGE_RESHAPE: tuple[int, int] = (
    CONF['IMAGE_HEIGHT_TO_MODEL'],
    CONF['IMAGE_WIDTH_TO_MODEL']
)
# _______________________________________________________


Redis_Client = redis.StrictRedis(
    host=CONF['REDIS_HOST'],
    port=CONF['REDIS_PORT'],
    db=CONF['REDIS_DB']
)


def Transform_Image(image: Image) -> Image:
    transform = transforms.Compose([
        transforms.Resize(IMAGE_RESHAPE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


def Image_Source(base64_encoded_image: str) -> np.ndarray:
    if 'base64' in base64_encoded_image:
        base64_encoded_image = base64_encoded_image.split('base64')[1][1:]

    image_bytes = b64decode(s=base64_encoded_image)
    io_img = BytesIO(image_bytes)
    image = Image.open(fp=io_img)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = Transform_Image(image=image)
    image = image.numpy()
    return image


def Analyze_Images() -> None:
    ov_core = Core()
    model = ov_core.read_model(model=MODEL_XML_PATH)
    compiledModel = ov_core.compile_model(
        model=model,
        device_name=MODEL_DEVICE)

    while True:
        batchSize: int = min(Redis_Client.llen(
            IMAGE_BATCH_NAME), IMAGE_BATCH_SIZE)

        if batchSize < IMAGE_BATCH_SIZE:
            print('Image Queue Len on Redis:', batchSize)
            time.sleep(1)
            continue

        images: list[str] = []
        imageUUIDs: list[str] = []
        with Redis_Client.pipeline() as pipe:
            for _ in range(batchSize):
                data = pipe.lpop(IMAGE_BATCH_NAME).execute()[0]
                uuid, image = json.loads(data)

                imageUUIDs.append(uuid)
                images.append(Image_Source(base64_encoded_image=image))

        print('* _________________________________________________________ *')
        startTime = time.monotonic()
        stackedImages: np.ndarray = np.stack(images, axis=0)
        outputs = compiledModel(stackedImages)[0]
        endTime = time.monotonic()
        processTime = '{:.2f} ms'.format(1000 * (endTime - startTime))
        print('Detect Time:', processTime)
        print('* _________________________________________________________ *')

        for i, output in enumerate(outputs):
            top1_index = np.argmax(output)
            top1_score = output[top1_index]
            top1_class_name = "None"
            if IMAGE_SIMILARITY_K < top1_score:
                top1_class_name = CLASS_NAMES[top1_index]

            imageResult = {
                'duration_ms': processTime,
                'result': CLASS_NAMES_WITH_IDS.get(top1_class_name, -1),
            }

            Redis_Client.setex(
                name=imageUUIDs[i],
                time=IMAGE_WAIT_TTL,
                value=json.dumps(imageResult))


if __name__ == "__main__":
    Analyze_Images()
