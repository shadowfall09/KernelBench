"""Reviewed scan-family hidden cases. Importing this module loads no tensor SDK."""

PROTOCOL = "kernelbench_pro_scan_cases_v1"
TASKS = (89, 90, 91, 92, 93)


def cases(pid, *, seed):
    """Preserve scan-axis 1, dtype roles and product signal; never resize timing."""
    if type(pid) is not int or pid not in TASKS or type(seed) is not int:
        raise ValueError("Reviewed scan inputs require a supported task and integer seed")
    import torch

    generator = torch.Generator().manual_seed(seed)
    result = []
    for label, shape in (("odd", (3, 17)), ("strided", (3, 17)), ("rank3", (2, 7, 5)),
                         ("singleton", (5, 1)), ("empty_axis", (3, 0))):
        values = torch.randint(-4, 5, shape, generator=generator).float() / 16
        if pid == 90:
            # Long random products collapse towards zero. These short products
            # stay informative while testing sign, scan direction and indexing.
            values = 1 + values / 4
            if shape[1]:
                values.select(1, 0).neg_()
        mask = torch.rand(shape, generator=generator) > 0.5
        if pid == 93:
            mask.select(0, 0).zero_()
            mask.select(0, -1).fill_(True)
        if label == "strided":
            backing = torch.empty((*shape[:-1], shape[-1] * 2), dtype=values.dtype)
            values = backing[..., ::2].copy_(values)
            mask_backing = torch.empty((*shape[:-1], shape[-1] * 2), dtype=torch.bool)
            mask = mask_backing[..., ::2].copy_(mask)
        result.append({"label": label, "inputs": [values, mask] if pid == 93 else [values],
                       "scan_axis": 1, "protocol": PROTOCOL})
    return result
