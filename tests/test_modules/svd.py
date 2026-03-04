import numpy as np


def generate(items: list[int]) -> list[np.ndarray]:
    return [np.random.rand(n, n) for n in items]


total = 0


def compute_svd_rank(A: list[np.ndarray]) -> list[int]:
    global total
    total += len(A)

    def svd_one(M: np.ndarray) -> int:
        S = np.linalg.svd(M, compute_uv=False)
        tol = np.max(M.shape) * np.spacing(S.max())
        return int(np.sum(S > tol))  # rank

    ret = [svd_one(M) for M in A]
    return ret


if __name__ == "__main__":
    import time

    start = time.time()
    DIM = 100
    N = 1000
    l1 = [generate([DIM]) for _ in range(N)]
    l2 = [compute_svd_rank(item) for item in l1]
    end = time.time()
    print(end - start, " seconds elapsed")
