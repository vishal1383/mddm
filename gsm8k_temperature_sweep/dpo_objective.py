"""Pure helpers for reward-ranked offline trajectory preferences."""
from __future__ import annotations

from typing import Sequence


DPO_SCHEMA_VERSION = "gsm8k_unmasking_dpo_v1"


def multiplicative_reward(correct: bool, nfe: int, length: int, alpha: float) -> float:
    """Apple's correctness-times-compute reward used to rank offline paths."""
    if not correct:
        return 0.0
    if length <= 0 or nfe <= 0 or alpha < 0:
        raise ValueError("length/nfe must be positive and alpha nonnegative")
    return ((length - min(nfe, length) + 1) / length) ** alpha


def strict_preference_pairs(rewards: Sequence[float], tolerance: float = 1e-12) -> list[tuple[int, int]]:
    """Return every strict within-prompt (chosen, rejected) reward ordering."""
    pairs: list[tuple[int, int]] = []
    for left in range(len(rewards)):
        for right in range(left + 1, len(rewards)):
            gap = float(rewards[left]) - float(rewards[right])
            if abs(gap) <= tolerance:
                continue
            pairs.append((left, right) if gap > 0 else (right, left))
    return pairs
