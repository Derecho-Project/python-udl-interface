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

    outputs = [None] * len(pairs)
    groups = {}

    for i, (a_capsule, b_capsule) in enumerate(pairs):
        a = torch.utils.dlpack.from_dlpack(a_capsule)
        b = torch.utils.dlpack.from_dlpack(b_capsule)
        key = (tuple(a.shape), tuple(b.shape), a.dtype, b.dtype, a.device)
        groups.setdefault(key, []).append((i, a, b))

    for bucket in groups.values():
        a_batch = torch.stack([a for _, a, _ in bucket], dim=0)
        b_batch = torch.stack([b for _, _, b in bucket], dim=0)
        c_batch = torch.matmul(a_batch, b_batch)
        batch_out = c_batch.detach().cpu().tolist()
        for (idx, _, _), out in zip(bucket, batch_out):
            outputs[idx] = out

    return outputs


def dense_matmul_from_dlpack_batch_sum(items):
    pairs = list(items)
    if not pairs:
        return []

    outputs = [None] * len(pairs)
    groups = {}

    for i, (a_capsule, b_capsule) in enumerate(pairs):
        a = torch.utils.dlpack.from_dlpack(a_capsule)
        b = torch.utils.dlpack.from_dlpack(b_capsule)
        key = (tuple(a.shape), tuple(b.shape), a.dtype, b.dtype, a.device)
        groups.setdefault(key, []).append((i, a, b))

    for bucket in groups.values():
        a_batch = torch.stack([a for _, a, _ in bucket], dim=0)
        b_batch = torch.stack([b for _, _, b in bucket], dim=0)
        c_batch = torch.matmul(a_batch, b_batch)
        per_item_sum = c_batch.sum(dim=(1, 2)).detach().cpu().tolist()
        for (idx, _, _), out in zip(bucket, per_item_sum):
            outputs[idx] = out

    return outputs
