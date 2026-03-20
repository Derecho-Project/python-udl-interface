import time


def invoke(items):
    if not items:
        return []

    delay_s = 5.0
    first = items[0]
    if isinstance(first, tuple) and len(first) == 2:
        delay_s = float(first[1])
        payload = [item[0] for item in items]
    else:
        payload = items

    time.sleep(delay_s)
    return payload
