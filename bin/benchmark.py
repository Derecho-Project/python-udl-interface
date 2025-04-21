import os
import csv
import time
import torch
import torch.utils.dlpack as dlpack
from FlagEmbedding import FlagModel
from concurrent.futures import ThreadPoolExecutor


MAX_WORKERS = 16
BATCH_SIZE = 128
model = FlagModel("BAAI/bge-small-en-v1.5", use_fp16=False, device="cuda:0", convert_to_numpy=False)
 
def invoke(queries: list[str]) -> torch.Tensor:
    embeddings = model.encode(queries)
    assert isinstance(embeddings, torch.Tensor)
    return embeddings

def split_into_batches(arr, n):
    return (arr[i:i + n] for i in range(0, len(arr), n))

def main():
    items = [
        "hello world"
    ] * 5000

    time_start = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks to the thread pool
        submission_times = []
        receive_times = []
        futures = []
        
        for batch in split_into_batches(items, BATCH_SIZE):
            submission_times.append(time.time())
            futures.append(executor.submit(invoke, batch))

        # Retrieve results as they become available
        for future in futures:
            future.result()
            receive_times.append(time.time())
            print("done")

    with open("csv/python_benchmark.csv", "w") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Queries", "Batch Size", "Num Workers", "Time (ms)"])
        for i in range(len(submission_times)):
            csv_writer.writerow([len(items), BATCH_SIZE, MAX_WORKERS, (receive_times[i] - submission_times[i]) * 1000])

if __name__ == "__main__":
    main()