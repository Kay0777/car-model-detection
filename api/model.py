from openvino.runtime import Core, CompiledModel, Model, AsyncInferQueue
from openvino.runtime.ie_api import InferRequest

from typing import Union, Any
import numpy as np


class TensorModel:
    def __init__(self, model: str,  device: str) -> None:
        self.__taskQueueSize: int = 1

        # Create an OpenVino Core
        self.openVinoCore: Core = Core()

        # Loading Model to Core
        self.model: Model = self.openVinoCore.read_model(model=model)
        self.config = {"PERFORMANCE_HINT": "THROUGHPUT"}

        # Create an Compiled Model from Core
        self.__compiledModel: CompiledModel = self.openVinoCore.compile_model(
            model=self.model,
            device_name=device,
            config = self.config)

        # Create an Infer Requester
        self.__asyncInferQueue: AsyncInferQueue = AsyncInferQueue(
            model=self.__compiledModel,
            jobs=self.__taskQueueSize)
        self.__asyncInferQueue.set_callback(self.__get_results)

    def __get_results(self, infer_request: InferRequest, _: Any) -> None:
        # Check if the inference is complete and handle the results
        self.__results = infer_request.results[self.__compiledModel.output(0)]

    @property
    def isAsyncInferQueueReady(self):
        return self.__asyncInferQueue.is_ready()

    def detect(self, inTenSorData: np.ndarray, name: Union[int, str] = 0) -> np.ndarray:
        # Start Asynchronously detection
        while not self.isAsyncInferQueueReady:
            pass

        # Async Start Analyze by Model
        self.__asyncInferQueue.start_async(inputs={name: inTenSorData})

        # Wait till task is done For GIL is not blocked!
        self.__asyncInferQueue.wait_all()

        # Return done results list [Tensors]
        return self.__results
