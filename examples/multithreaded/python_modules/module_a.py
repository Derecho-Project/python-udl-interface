import time


def invoke(durations: list[float]) -> list[float]:
    """Simulate an expensive operation by sleeping.

    time.sleep() releases the GIL, allowing other Python threads to run
    concurrently. Returns the actual elapsed time for each item.
    """
    results = []
    for seconds in durations:
        start = time.monotonic()
        time.sleep(seconds)
        results.append(time.monotonic() - start)
    return results
