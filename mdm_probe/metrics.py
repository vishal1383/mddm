from __future__ import annotations

from .types import ConfidenceResult


def remaining_positions(T: int, anchors: set[int] | list[int] | tuple[int, ...]) -> list[int]:
    anchor_set = set(anchors)
    return [pos for pos in range(T) if pos not in anchor_set]


def score_p_gt(
    p_gt: list[float],
    anchors: set[int] | list[int] | tuple[int, ...],
    *,
    reduction: str = "mean",
) -> float:
    return score_values(p_gt, anchors, reduction=reduction)


def score_values(
    values_by_position: list[float],
    anchors: set[int] | list[int] | tuple[int, ...],
    *,
    reduction: str = "mean",
) -> float:
    values = [
        values_by_position[pos]
        for pos in remaining_positions(len(values_by_position), anchors)
    ]
    if not values:
        return 0.0
    if reduction == "sum":
        return float(sum(values))
    if reduction == "mean":
        return float(sum(values) / len(values))
    raise ValueError("reduction must be 'sum' or 'mean'")


def p_gt_gain(
    before: ConfidenceResult,
    after: ConfidenceResult,
    *,
    exclude: set[int] | list[int] | tuple[int, ...],
) -> tuple[float, list[float]]:
    return value_gain(before.p_gt, after.p_gt, exclude=exclude)


def max_p_gain(
    before: ConfidenceResult,
    after: ConfidenceResult,
    *,
    exclude: set[int] | list[int] | tuple[int, ...],
) -> tuple[float, list[float]]:
    return value_gain(before.max_p, after.max_p, exclude=exclude)


def value_gain(
    before: list[float],
    after: list[float],
    *,
    exclude: set[int] | list[int] | tuple[int, ...],
) -> tuple[float, list[float]]:
    positions = _target_positions(len(before), exclude)
    deltas = [after[pos] - before[pos] for pos in positions]
    return float(sum(deltas)), [float(v) for v in deltas]


def information_gain(
    before: ConfidenceResult,
    after: ConfidenceResult,
    *,
    exclude: set[int] | list[int] | tuple[int, ...],
) -> tuple[float, list[float]]:
    """Sum_q H_before(q) - H_after(q) over still-masked target positions."""

    positions = _target_positions(len(before.entropy), exclude)
    gains = [before.entropy[pos] - after.entropy[pos] for pos in positions]
    return float(sum(gains)), [float(v) for v in gains]


def _target_positions(T: int, exclude: set[int] | list[int] | tuple[int, ...]) -> list[int]:
    exclude_set = set(exclude)
    return [pos for pos in range(T) if pos not in exclude_set]
