import torch.nn.functional as F
import numpy as np
import torch
import time

from PIL import Image
from typing import Union
from collections import deque
import multiprocessing as mp


# from torch2trt import TRTModule

from model import TensorModel

from utils import (
    ClassesAndClassIDs,
    CreateImageUUIDName,
    ConvertBase64ImageToTorchImageSource,
    ConvertBase64ImageToOpenVinoImageSource)

from config import CONF

MANAGER = mp.Manager()

RESULTS = MANAGER.dict()
TASKS: deque = deque()
SHARED_TASKS: mp.Queue = mp.Queue()


def runAnalysis():
    if CONF['ANALYZE_WITH_OPENVINO']:
        Analysis_With_OpenVino()
    else:
        Analysis_With_Torch()


def PreProcessing(base64_encoded_image: str) -> tuple[str, bool]:
    global TASKS, SHARED_TASKS

    # # Generate a unique identifier
    # if len(TASKS) > CONF['COUNT_OF_TASKS']:
    #     return '', True

    # Generate a unique identifier
    if SHARED_TASKS.qsize() > CONF['COUNT_OF_TASKS']:
        return '', True

    # Convert Base64 Image To IO Buffer Image As Pillow Image
    if CONF['ANALYZE_WITH_OPENVINO']:
        image: Image = ConvertBase64ImageToOpenVinoImageSource(
            base64_encoded_image=base64_encoded_image)
    else:
        image: Image = ConvertBase64ImageToTorchImageSource(
            base64_encoded_image=base64_encoded_image)

    print('PreProcessing', id(TASKS))
    # Create Image UUID
    imageUUID: str = CreateImageUUIDName()

    # # Add Image To Task
    # TASKS.append((image, imageUUID))

    # Add Image To Task
    SHARED_TASKS.put((image, imageUUID))
    print('PreProcessing', SHARED_TASKS.qsize())

    return imageUUID, False


def ProcessedImageResult(imageUUID: str) -> Union[dict, None]:
    global RESULTS
    result = RESULTS.get(imageUUID, None)
    return result


def Analysis_With_Torch() -> None:
    global TASKS, RESULTS

    classNames, classNamesWithIDs = ClassesAndClassIDs(
        classFilename=CONF['CLASS_NAMES'],
        classIDsFilename=CONF['CLASS_NAMES_WITH_IDS'])

    print('Car Torch model is loading...')
    device = torch.device("cuda:0")
    model: TRTModule = TRTModule()  # type: ignore
    model.load_state_dict(torch.load(f=CONF['CAR_TORCH_MODEL']))
    model = model.to(device)
    print('Car Torch model is loaded!')

    while True:
        if len(TASKS) < CONF['COUNT_OF_PROCCESSES_IMAGES']:
            continue

        batches: list = []
        uuids: list = []
        for _ in range(CONF['COUNT_OF_PROCCESSES_IMAGES']):
            image, imageUUID = TASKS.pop()

            batches.append(image)
            uuids.append(imageUUID)

        batches = torch.cat(tensors=batches, dim=0).to(device)
        startTime = time.monotonic()
        with torch.no_grad():
            outputs = model(batches)
            endTime = time.monotonic()
            analysisTime = round(1000 * (endTime - startTime))

            for i, output in enumerate(outputs):
                probabilities = torch.nn.functional.softmax(output, dim=0)

                scoreTop1, indexTop1 = torch.max(probabilities, 0)

                classNameTop1 = "None"
                if scoreTop1 >= CONF['UPPER_SCORE']:
                    classNameTop1 = classNames[indexTop1]

                RESULTS[uuids[i]] = {
                    'duration_ms': analysisTime,
                    'result': classNamesWithIDs.get(classNameTop1, -1),
                    'prediction': {
                        classNamesWithIDs.get(className, -1): round(prob.item() * 100, 2)
                        for className, prob in zip(classNames, probabilities)
                    }
                }


def Analysis_With_OpenVino() -> None:
    global TASKS, RESULTS, SHARED_TASKS

    classNames, classNamesWithIDs = ClassesAndClassIDs(
        classFilename=CONF['CLASS_NAMES'],
        classIDsFilename=CONF['CLASS_NAMES_WITH_IDS'])

    print('Car Tensor model is loading....')
    carModel: TensorModel = TensorModel(
        model=CONF['CAR_OPENVINO_MODEL'],
        device=CONF['DEVICE'])
    print('Car Tensor model is loaded!')
    # print('Analysis_With_OpenVino', id(TASKS))
    print('Analysis_With_OpenVino', id(SHARED_TASKS))

    while True:
        if SHARED_TASKS.qsize() == 0:
            continue
        else:
            print('Analysis_With_OpenVino', SHARED_TASKS.qsize())

        # if len(TASKS) == 0:
        #     continue
        # else:
        #     print('Analysis_With_OpenVino', len(TASKS))

        batches: list = []
        uuids: list = []
        # minBatchCount: int = min(len(TASKS), CONF['COUNT_OF_PROCCESSES_IMAGES'])
        minBatchCount: int = min(SHARED_TASKS.qsize(),
                                 CONF['COUNT_OF_PROCCESSES_IMAGES'])
        for _ in range(minBatchCount):
            # image, imageUUID = TASKS.pop()
            image, imageUUID = SHARED_TASKS.get()

            batches.append(image)
            uuids.append(imageUUID)

        startTime = time.monotonic()
        batches = np.squeeze(np.stack(batches, axis=0), axis=1)
        outputs = carModel.detect(inTenSorData=batches)
        endTime = time.monotonic()
        analysisTime = round(1000 * (endTime - startTime))
        print("Time:", analysisTime)
        for i, output in enumerate(outputs):
            # Apply softmax along the correct dimension
            probabilities = F.softmax(torch.tensor(output), dim=0).numpy()
            indexTop1 = np.argmax(probabilities)
            scoreTop1 = probabilities[indexTop1]
            classNameTop1 = "None"
            if scoreTop1 >= CONF['UPPER_SCORE']:
                classNameTop1 = classNames[indexTop1]

            RESULTS[uuids[i]] = {
                'duration_ms': analysisTime,
                'result': classNamesWithIDs.get(classNameTop1, -1),
                'prediction': {
                    classNamesWithIDs.get(className, -1): round(prob.item() * 100, 2)
                    for className, prob in zip(classNames, probabilities)
                }
            }
