import time
import torch
import torch.utils.dlpack as dlpack

def invoke(a, b):
    tensor = torch.rand((a, b)).to("cuda:0")
    return dlpack.to_dlpack(tensor) # type: ignore
