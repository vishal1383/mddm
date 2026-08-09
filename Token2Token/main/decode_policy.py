"""Shared catalyst-selection rules used by decoding and target generation."""
from __future__ import annotations

import torch


def joint_topk_catalyst_mask(
    confidence,
    allowed,
    budget,
    *,
    additional_min_confidence=0.0,
    additional_min_ratio=0.0,
):
    """Select up to ``budget`` positions jointly from one model forward.

    The highest-confidence allowed position is always retained. Gates apply
    only to additional positions, relative to that first choice.
    """
    if confidence.ndim != 2 or allowed.ndim != 2:
        raise ValueError("confidence and allowed must be rank-two tensors")
    if confidence.shape != allowed.shape:
        raise ValueError("confidence and allowed must have matching shapes")
    if allowed.dtype != torch.bool:
        raise ValueError("allowed must be a boolean tensor")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0.0 <= additional_min_confidence < 1.0:
        raise ValueError("additional minimum confidence must be in [0, 1)")
    if not 0.0 <= additional_min_ratio <= 1.0:
        raise ValueError("additional minimum ratio must be in [0, 1]")

    width = confidence.shape[1]
    if width == 0:
        return torch.zeros_like(allowed)
    count = min(int(budget), width)
    scores = confidence.masked_fill(~allowed, -torch.inf)
    values, indices = scores.topk(count, dim=1)
    accepted = torch.isfinite(values)
    if count > 1:
        accepted[:, 1:] &= values[:, 1:].ge(additional_min_confidence)
        accepted[:, 1:] &= values[:, 1:].ge(
            values[:, :1] * additional_min_ratio
        )

    chosen = torch.zeros_like(allowed)
    chosen.scatter_(1, indices, accepted)
    return chosen & allowed
