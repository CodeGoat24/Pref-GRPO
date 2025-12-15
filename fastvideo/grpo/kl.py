from contextlib import contextmanager
from typing import Iterator

import torch


def compute_kl_loss(prev_sample_mean, prev_sample_mean_ref, noise_scale_ref):
    diff_sq = (prev_sample_mean.to(torch.float32) - prev_sample_mean_ref.to(torch.float32)) ** 2
    reduce_dims = tuple(range(1, diff_sq.ndim))
    denom = 2.0 * (noise_scale_ref.to(torch.float32) ** 2)
    denom = torch.clamp(denom, min=1e-20)
    per_sample = diff_sq.mean(dim=reduce_dims) / denom
    return per_sample.mean()


@contextmanager
def disable_lora_adapters(model) -> Iterator[None]:
    """
    Best-effort adapter disabling across PEFT and diffusers adapter mixins.
    Handles wrappers like FSDP/DDP by attempting to operate on `model.module` when present.
    """
    underlying = getattr(model, "module", model)

    if hasattr(underlying, "disable_adapter") and callable(getattr(underlying, "disable_adapter")):
        with underlying.disable_adapter():
            yield
        return

    if hasattr(underlying, "disable_adapters") and callable(getattr(underlying, "disable_adapters")):
        try:
            maybe_cm = underlying.disable_adapters()
        except TypeError:
            maybe_cm = None
        if maybe_cm is not None and hasattr(maybe_cm, "__enter__") and hasattr(maybe_cm, "__exit__"):
            with maybe_cm:
                yield
            return

        try:
            underlying.disable_adapters()
            yield
        finally:
            try:
                if hasattr(underlying, "enable_adapters"):
                    underlying.enable_adapters()
            except Exception:
                pass
        return

    yield

