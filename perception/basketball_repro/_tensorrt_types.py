"""Interfaces used at the TensorRT compiled-extension boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class ExecutionContext(Protocol):
    def set_tensor_address(self, name: str, memory: int) -> bool: ...

    def execute_async_v3(self, stream_handle: int) -> bool: ...


class Engine(Protocol):
    num_io_tensors: int

    def create_execution_context(self) -> ExecutionContext: ...

    def get_tensor_name(self, index: int) -> str: ...

    def get_tensor_shape(self, name: str) -> Sequence[int]: ...

    def get_tensor_dtype(self, name: str) -> object: ...

    def get_tensor_mode(self, name: str) -> object: ...


class RuntimeProtocol(Protocol):
    def deserialize_cuda_engine(self, serialized_engine: bytes) -> Engine | None: ...


class LoggerFactory(Protocol):
    WARNING: object

    def __call__(self, severity: object) -> object: ...


class TensorIOModeProtocol(Protocol):
    INPUT: object


class TensorRT(Protocol):
    Logger: LoggerFactory
    TensorIOMode: TensorIOModeProtocol

    def Runtime(self, logger: object) -> RuntimeProtocol: ...

    def nptype(self, dtype: object) -> type[np.generic]: ...
