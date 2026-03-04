def add(A: list[tuple[int, int]]) -> list[int]:
    ret = [t[0] + t[1] for t in A]
    return ret
