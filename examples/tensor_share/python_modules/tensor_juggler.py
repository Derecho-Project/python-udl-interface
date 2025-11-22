import torch.utils.dlpack as dlpack
import torch
def generate_tensor(length: int, width: int):
    tensor = torch.rand(length, width, dtype=torch.float32)
    return dlpack.to_dlpack(tensor) # type: ignore

def multiply_sum_tensors(a, b) -> int:
    a_tensor = dlpack.from_dlpack(a)
    b_tensor = dlpack.from_dlpack(b)
    c_tensor = a_tensor * b_tensor
    d_tensor = torch.sum(c_tensor)
    return int(d_tensor.item())