import torch


def has_cuda() -> bool:
    return bool(torch.cuda.is_available())


def dense_matmul_from_dlpack(a_capsule, b_capsule):
    a = torch.utils.dlpack.from_dlpack(a_capsule)
    b = torch.utils.dlpack.from_dlpack(b_capsule)
    c = torch.matmul(a, b)
    return c.detach().cpu().tolist()


def dense_matmul_from_dlpack_batch(items):
    pairs = list(items)
    if not pairs:
        return []

    a_tensors = [torch.utils.dlpack.from_dlpack(a_capsule) for a_capsule, _ in pairs]
    b_tensors = [torch.utils.dlpack.from_dlpack(b_capsule) for _, b_capsule in pairs]

    # Execute one batched matmul on GPU: [N, M, K] x [N, K, P] -> [N, M, P].
    a_batch = torch.stack(a_tensors, dim=0)
    b_batch = torch.stack(b_tensors, dim=0)
    c_batch = torch.matmul(a_batch, b_batch)
    return c_batch.detach().cpu().tolist()


def dense_matmul_from_dlpack_batch_sum(items):
    pairs = list(items)
    if not pairs:
        return []

    a_tensors = [torch.utils.dlpack.from_dlpack(a_capsule) for a_capsule, _ in pairs]
    b_tensors = [torch.utils.dlpack.from_dlpack(b_capsule) for _, b_capsule in pairs]

    # Batched GPU matmul, then reduce each output matrix on GPU to one scalar.
    a_batch = torch.stack(a_tensors, dim=0)
    b_batch = torch.stack(b_tensors, dim=0)
    c_batch = torch.matmul(a_batch, b_batch)
    per_item_sum = c_batch.sum(dim=(1, 2))
    return per_item_sum.detach().cpu().tolist()
