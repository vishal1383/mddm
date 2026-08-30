"""Correctness-first preference construction for pure trajectory DPO."""
from __future__ import annotations

from typing import Sequence


DPO_SCHEMA_VERSION = "gsm8k_unmasking_dpo_v3"


def frontier_preference_pairs(
    correctness: Sequence[bool], nfes: Sequence[int]
) -> tuple[int | None, list[tuple[int, int, str]]]:
    """Build at most one safety pair and one efficiency pair per prompt.

    The fastest correct trajectory is the winner.  It is compared with the
    fastest incorrect trajectory (the tempting unsafe alternative) and with
    the slowest correct trajectory (the conservative alternative).  Wrong
    trajectories are never preferred merely for being fast, and prompts with
    no correct trajectory fabricate no signal.
    """

    if len(correctness) != len(nfes) or not correctness:
        raise ValueError("correctness and NFE lists must be nonempty and aligned")
    if any(int(nfe) <= 0 for nfe in nfes):
        raise ValueError("every trajectory must have positive NFE")
    correct = [index for index, value in enumerate(correctness) if bool(value)]
    if not correct:
        return None, []
    winner = min(correct, key=lambda index: (int(nfes[index]), index))
    pairs: list[tuple[int, int, str]] = []

    incorrect = [index for index, value in enumerate(correctness) if not bool(value)]
    if incorrect:
        hard_negative = min(incorrect, key=lambda index: (int(nfes[index]), index))
        pairs.append((winner, hard_negative, "safety"))

    slow_correct = max(correct, key=lambda index: (int(nfes[index]), -index))
    if int(nfes[slow_correct]) > int(nfes[winner]):
        pairs.append((winner, slow_correct, "efficiency"))
    return winner, pairs
