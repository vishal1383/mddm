from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Target:
    index: int
    token_id: int
    token_text: str
    gold_position: int


@dataclass(frozen=True)
class Commit:
    target_index: int
    canvas_position: int
    confidence: float


@dataclass(frozen=True)
class Losses:
    total: torch.Tensor
    placement: torch.Tensor
    order: torch.Tensor
    grounding: torch.Tensor


def select_targets(
    token_ids: Sequence[int],
    tokenizer,
    *,
    count: int = 5,
    max_right_fraction: float = 0.75,
    policy: str = "spread",
    seed: int = 0,
    mask_token_id: int | None = None,
) -> list[Target]:
    """Choose non-whitespace targets without using the far-right completion tail."""

    if count <= 0 or not token_ids:
        return []
    if not 0 < max_right_fraction <= 1:
        raise ValueError("max_right_fraction must be in (0, 1]")
    upper = math.floor((len(token_ids) - 1) * max_right_fraction)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    candidates = []
    for position, token_id in enumerate(token_ids):
        if position > upper or token_id == mask_token_id or token_id in special:
            continue
        text = str(tokenizer.decode([int(token_id)]))
        if text.strip():
            candidates.append((position, int(token_id), text))
    if not candidates:
        return []

    count = min(count, len(candidates))
    if policy == "spread":
        if count == 1:
            selected = [candidates[len(candidates) // 2]]
        else:
            indices = {
                round(index * (len(candidates) - 1) / (count - 1))
                for index in range(count)
            }
            indices.update(index for index in range(len(candidates)) if len(indices) < count)
            selected = [candidates[index] for index in sorted(indices)[:count]]
    elif policy == "prefix":
        selected = candidates[:count]
    elif policy == "random":
        selected = sorted(random.Random(seed).sample(candidates, count))
    else:
        raise ValueError("policy must be spread, prefix, or random")
    return [
        Target(index, token_id, text, gold_position)
        for index, (gold_position, token_id, text) in enumerate(selected)
    ]


def make_canvas(
    length: int,
    mask_token_id: int,
    targets: Sequence[Target],
    commits: dict[int, Commit],
) -> list[int]:
    canvas = [int(mask_token_id)] * length
    by_index = {target.index: target for target in targets}
    for target_index, commit in commits.items():
        canvas[commit.canvas_position] = by_index[target_index].token_id
    return canvas


def position_priors(
    targets: Sequence[Target],
    commits: dict[int, Commit],
    length: int,
    *,
    sigma: float,
    max_right_fraction: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Make relative-order-aware Gaussian priors for target placement."""

    if length <= 0:
        raise ValueError("length must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0 < max_right_fraction <= 1:
        raise ValueError("max_right_fraction must be in (0, 1]")
    if not targets:
        return torch.empty((0, length), device=device)

    right_limit = math.floor((length - 1) * max_right_fraction)
    occupied = {commit.canvas_position for commit in commits.values()}
    blocked = torch.zeros(length, device=device, dtype=torch.bool)
    if occupied:
        blocked[list(occupied)] = True
    committed_targets = [
        (target, commits[target.index])
        for target in targets
        if target.index in commits
    ]
    rows = []
    positions = torch.arange(length, device=device, dtype=torch.float32)
    for target in targets:
        if target.index in commits:
            row = torch.zeros(length, device=device)
            row[commits[target.index].canvas_position] = 1.0
            rows.append(row)
            continue

        left = [item for item in committed_targets if item[0].gold_position < target.gold_position]
        right = [item for item in committed_targets if item[0].gold_position > target.gold_position]
        left_neighbor = max(left, key=lambda item: item[0].gold_position) if left else None
        right_neighbor = min(right, key=lambda item: item[0].gold_position) if right else None
        center = _mapped_center(target, left_neighbor, right_neighbor)

        allowed = positions <= right_limit
        if occupied:
            allowed &= ~blocked
        if not bool(allowed.any()):
            raise ValueError("no free position remains inside max_right_fraction")
        center = min(max(center, 0.0), float(right_limit))
        weights = torch.exp(-0.5 * ((positions - center) / sigma) ** 2)
        weights *= allowed.float()
        rows.append(weights / weights.sum().clamp_min(1e-12))
    return torch.stack(rows)


def anchor_loss(
    logits: torch.Tensor,
    targets: Sequence[Target],
    commits: dict[int, Commit],
    priors: torch.Tensor,
    *,
    lambda_order: float = 0.25,
    lambda_ce: float = 1.0,
) -> tuple[Losses, torch.Tensor, torch.Tensor]:
    """Placement + relative order + token identity grounding loss."""

    active = [target for target in targets if target.index not in commits]
    if not active:
        raise ValueError("anchor_loss requires at least one uncommitted target")
    indices = torch.tensor([target.index for target in active], device=logits.device)
    token_ids = torch.tensor([target.token_id for target in active], device=logits.device)
    gold_positions = torch.tensor(
        [target.gold_position for target in active], device=logits.device
    )
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs[:, token_ids].transpose(0, 1)
    scores = token_log_probs + torch.log(priors[indices].clamp_min(1e-12))
    evidence = torch.logsumexp(scores, dim=-1)
    q_active = torch.softmax(scores, dim=-1)
    placement = -evidence.mean()
    grounding = F.cross_entropy(logits[gold_positions].float(), token_ids)

    q_by_index = {target.index: q_active[row] for row, target in enumerate(active)}
    all_q = []
    for target in targets:
        if target.index in commits:
            row = torch.zeros(logits.shape[0], device=logits.device)
            row[commits[target.index].canvas_position] = 1.0
            all_q.append(row)
        else:
            all_q.append(q_by_index[target.index])
    order = relative_order_loss(
        torch.stack(all_q),
        torch.tensor([target.gold_position for target in targets], device=logits.device),
    )
    total = placement + lambda_order * order + lambda_ce * grounding
    return Losses(total, placement, order, grounding), q_active, evidence


def relative_order_loss(q: torch.Tensor, gold_positions: torch.Tensor) -> torch.Tensor:
    if q.shape[0] < 2:
        return q.sum() * 0.0
    order = torch.argsort(gold_positions)
    losses = []
    for pair_index in range(len(order) - 1):
        left = q[int(order[pair_index])]
        right = q[int(order[pair_index + 1])]
        left_mass_before = torch.cumsum(left, dim=-1) - left
        probability = (left_mass_before * right).sum()
        losses.append(-torch.log(probability.clamp_min(1e-12)))
    return torch.stack(losses).mean()


def _mapped_center(target, left_neighbor, right_neighbor) -> float:
    if left_neighbor and right_neighbor:
        left_target, left_commit = left_neighbor
        right_target, right_commit = right_neighbor
        ratio = (target.gold_position - left_target.gold_position) / max(
            1, right_target.gold_position - left_target.gold_position
        )
        return left_commit.canvas_position + ratio * (
            right_commit.canvas_position - left_commit.canvas_position
        )
    if left_neighbor:
        left_target, left_commit = left_neighbor
        return left_commit.canvas_position + target.gold_position - left_target.gold_position
    if right_neighbor:
        right_target, right_commit = right_neighbor
        return right_commit.canvas_position - right_target.gold_position + target.gold_position
    return float(target.gold_position)
