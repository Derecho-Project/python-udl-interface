import torch.utils.dlpack as dlpack
import torch
import time
def generate_tensor(length: int, width: int):
    tensor = torch.rand(length, width, dtype=torch.float32)
    return dlpack.to_dlpack(tensor) # type: ignore

def multiply_sum_tensors(a, b) -> int:
    a_tensor = dlpack.from_dlpack(a)
    b_tensor = dlpack.from_dlpack(b)
    c_tensor = a_tensor * b_tensor
    d_tensor = torch.sum(c_tensor)
    return int(d_tensor.item())


def main():
    DIM = 100
    NUM_ITERATIONS = 600

    x = time.perf_counter_ns()
    a_tensors = [ generate_tensor(DIM, DIM) for _ in range(NUM_ITERATIONS) ]
    b_tensors = [ generate_tensor(DIM, DIM) for _ in range(NUM_ITERATIONS) ]

    for i in range(NUM_ITERATIONS):
        result = multiply_sum_tensors(a_tensors[i], b_tensors[i])
    y = time.perf_counter_ns()
    print(f"Elapsed Time (us): {(y - x) / 1e3}")

if __name__ == "__main__":
    main()