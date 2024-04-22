from torchvision import transforms
from io import BytesIO
from PIL import Image
from uuid import uuid4
import numpy as np
import base64
import os
import json


def ReadClassesFromFile(filename: str) -> list[str]:
    if not os.path.exists(path=filename):
        raise FileNotFoundError(f'File not found: {filename}')

    with open(file=filename, mode='r') as file:
        classes = [
            line.strip()
            for line in file.readlines()
        ]
        file.close()
    return classes


def ReadClassesWithIDsFromJsonFile(filename: str) -> dict[str, int]:
    if not os.path.exists(path=filename):
        raise FileNotFoundError(f'File not found: {filename}')

    with open(file=filename, mode='r') as file:
        classesWithIDs = json.loads(file.read())
        file.close()
    return classesWithIDs


def ClassesAndClassIDs(classFilename: str, classIDsFilename: str) -> tuple[list[str], dict[str, int]]:
    classNames: list[str] = ReadClassesFromFile(filename=classFilename)
    classNamesWithIDs: dict[str, int] = ReadClassesWithIDsFromJsonFile(
        filename=classIDsFilename)

    return classNames, classNamesWithIDs


def DeCode64ToBytes(source: str) -> bytes:
    if 'base64' in source:
        source = source.split('base64')[1][1:]
    image_bytes = base64.b64decode(s=source)
    return image_bytes


def CreateImageUUIDName() -> str:
    return uuid4().hex


def TransformImage(image: Image) -> Image:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


def ConvertBase64ImageToTorchImageSource(base64_encoded_image: str) -> Image:
    image_bytes = DeCode64ToBytes(source=base64_encoded_image)
    io_img = BytesIO(image_bytes)

    new_image = Image.open(fp=io_img)

    filename: str = os.path.join(os.path.dirname(__file__), "addition.jpg")
    if not os.path.exists(path=filename):
        new_image.save(fp=filename, format="JPEG")
    addition = Image.open(fp=filename)

    # Convert both images to RGB to avoid mode mismatch
    new_image = new_image.convert("RGB")
    addition = addition.convert("RGB")

    width, height = new_image.size
    addition = addition.resize(
        (width, int(addition.height * width / addition.width)))

    image = Image.new('RGB', (width, new_image.height + addition.height))
    image.paste(new_image, (0, 0))
    image.paste(addition, (0, new_image.height))

    # Apply the transformation and add batch dimension
    image: Image = TransformImage(image=image)
    image = image.unsqueeze(0)

    return image


def ConvertBase64ImageToOpenVinoImageSource(base64_encoded_image: str) -> Image:
    image_bytes = DeCode64ToBytes(source=base64_encoded_image)
    io_img = BytesIO(image_bytes)

    image = Image.open(fp=io_img)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = TransformImage(image=image)
    image = image.numpy()
    image = np.expand_dims(image, axis=0)

    return image
