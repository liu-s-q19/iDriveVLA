import torch


def masked_token_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Average token-level values with a completion mask, then average across batch."""
    values = values.to(dtype=torch.float32)
    mask = mask.to(device=values.device, dtype=torch.float32)
    denom = mask.sum(dim=1).clamp_min(eps)
    per_sample = (values * mask).sum(dim=1) / denom
    return per_sample.mean()
