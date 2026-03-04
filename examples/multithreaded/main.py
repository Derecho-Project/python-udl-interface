"""Python baseline for apples-to-apples comparison against the C++ version."""

import time

from python_modules.module_a import invoke as invoke_a
from python_modules.module_b import invoke as invoke_b

NUM_REQUESTS = 50
SLEEP_SECONDS = 0.01

start = time.monotonic()

# Sequential: call each module one item at a time, matching the C++ per-request granularity
results_a = [invoke_a([SLEEP_SECONDS])[0] for _ in range(NUM_REQUESTS)]
results_b = [invoke_b([SLEEP_SECONDS])[0] for _ in range(NUM_REQUESTS)]

elapsed_ms = (time.monotonic() - start) * 1000

expected_sequential = NUM_REQUESTS * 2 * SLEEP_SECONDS * 1000
print(f"Expected sequential: {expected_sequential:.0f} ms")
print(f"Actual:              {elapsed_ms:.0f} ms")
print(f"Speedup:             {expected_sequential / elapsed_ms:.2f}x")

# Write results to file for diffing
with open("/tmp/multithreaded_py.txt", "w") as f:
    for i, val in enumerate(results_a):
        f.write(f"a[{i}] = {val:.10g}\n")
    for i, val in enumerate(results_b):
        f.write(f"b[{i}] = {val:.10g}\n")

print("Results written to /tmp/multithreaded_py.txt")
