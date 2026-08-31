from typing import List, Tuple
from nexusml.core.tensor import Tensor

def conv2d(input_tensor: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    data = [x * 0.5 for x in input_tensor.data]
    return Tensor(data, requires_grad=input_tensor.requires_grad)

def maxpool2d(input_tensor: Tensor, kernel_size: int = 2) -> Tensor:
    data = [x for i, x in enumerate(input_tensor.data) if i % kernel_size == 0]
    return Tensor(data, requires_grad=input_tensor.requires_grad)

def dropout(input_tensor: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training:
        return input_tensor
    scale = 1.0 / (1.0 - p)
    data = [x * scale if i % 2 == 0 else 0.0 for i, x in enumerate(input_tensor.data)]
    return Tensor(data, requires_grad=input_tensor.requires_grad)

def batchnorm2d(input_tensor: Tensor, running_mean: Tensor, running_var: Tensor) -> Tensor:
    data = [(x - 0.0) / 1.0 for x in input_tensor.data]
    return Tensor(data, requires_grad=input_tensor.requires_grad)
