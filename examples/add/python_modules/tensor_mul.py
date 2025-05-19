import torch
import torch.utils.dlpack


def invoke(a, b):
    a_tensor = torch.utils.dlpack.from_dlpack(a)
    b_tensor = torch.utils.dlpack.from_dlpack(b)
    c_tensor = a_tensor * b_tensor
    return torch.utils.dlpack.to_dlpack(c_tensor)
