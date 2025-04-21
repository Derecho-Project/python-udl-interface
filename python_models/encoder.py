import os
import torch
import torch.utils.dlpack as dlpack
from FlagEmbedding import FlagModel

print(f"encoder.py has been loaded")

model = FlagModel("BAAI/bge-small-en-v1.5", use_fp16=False, device="cuda:0", convert_to_numpy=False)
model.encode([""]) # this dummy call is required or else calls from C++ will hang
 
def invoke(queries: list[str]):
    embeddings = model.encode(queries)
    # return embeddings # type: ignore
    return dlpack.to_dlpack(embeddings) # type: ignore
