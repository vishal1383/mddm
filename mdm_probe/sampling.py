from __future__ import annotations

from .anchors import layout_anchors
from .metrics import information_gain, p_gt_gain, remaining_positions, score_p_gt
from .types import ConfidenceResult, EncodedExample


def run_single_anchor_probe(
    model,
    encoded: EncodedExample,
    *,
    batch_size: int = 1,
    score_reduction: str = "mean",
) -> list[dict]:
    T = encoded.completion_length
    candidates = anchor_candidates(encoded)
    baseline = model.compute_p_gt_batch(encoded, [()], batch_size=batch_size)[0]
    anchored = model.compute_p_gt_batch(encoded, [(pos,) for pos in candidates], batch_size=batch_size)

    rows: list[dict] = []
    for pos, after in zip(candidates, anchored):
        targets = remaining_positions(T, [pos])
        ig_sum, ig_by_target = information_gain(baseline, after, exclude=[pos])
        p_gain_sum, p_gain_by_target = p_gt_gain(baseline, after, exclude=[pos])
        rows.append(
            {
                "example_id": encoded.example_id,
                "anchor_position": pos,
                "anchor_token_id": encoded.completion_token_ids[pos],
                "anchor_token_text": encoded.completion_token_texts[pos],
                "target_positions": targets,
                "information_gain": ig_sum,
                "information_gain_by_target": ig_by_target,
                "p_gt_gain": p_gain_sum,
                "p_gt_gain_by_target": p_gain_by_target,
                "score_before": score_p_gt(baseline.p_gt, [pos], reduction=score_reduction),
                "score_after": score_p_gt(after.p_gt, [pos], reduction=score_reduction),
                "p_gt_before": list(baseline.p_gt),
                "p_gt_after": list(after.p_gt),
                "entropy_before": list(baseline.entropy),
                "entropy_after": list(after.entropy),
            }
        )
    return rows


def run_greedy_k_probe(
    model,
    encoded: EncodedExample,
    *,
    max_k: int = 10,
    batch_size: int = 1,
    score_reduction: str = "mean",
) -> tuple[list[dict], list[dict]]:
    eligible = anchor_candidates(encoded)
    anchors: list[int] = []
    current = model.compute_p_gt_batch(encoded, [()], batch_size=batch_size)[0]
    step_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for step in range(1, min(max_k, len(eligible)) + 1):
        candidates = [pos for pos in eligible if pos not in anchors]
        candidate_sets = [tuple(sorted(anchors + [candidate])) for candidate in candidates]
        candidate_results = model.compute_p_gt_batch(encoded, candidate_sets, batch_size=batch_size)

        scored: list[tuple[float, int, ConfidenceResult, list[float], float, list[float]]] = []
        for candidate, after in zip(candidates, candidate_results):
            exclude = anchors + [candidate]
            ig_sum, ig_by_target = information_gain(current, after, exclude=exclude)
            p_gain_sum, p_gain_by_target = p_gt_gain(current, after, exclude=exclude)
            scored.append((ig_sum, candidate, after, ig_by_target, p_gain_sum, p_gain_by_target))
            candidate_rows.append(
                {
                    "example_id": encoded.example_id,
                    "step": step,
                    "current_anchors": list(anchors),
                    "candidate_position": candidate,
                    "candidate_token_id": encoded.completion_token_ids[candidate],
                    "candidate_token_text": encoded.completion_token_texts[candidate],
                    "information_gain": ig_sum,
                    "p_gt_gain": p_gain_sum,
                }
            )

        best_ig, best_pos, best_result, best_ig_by_target, best_p_gain, best_p_by_target = max(
            scored,
            key=lambda item: item[0],
        )
        previous = current
        anchors.append(best_pos)
        anchors.sort()
        current = best_result
        step_rows.append(
            {
                "example_id": encoded.example_id,
                "k": step,
                "selected_anchor_position": best_pos,
                "selected_anchor_token_id": encoded.completion_token_ids[best_pos],
                "selected_anchor_token_text": encoded.completion_token_texts[best_pos],
                "anchors": list(anchors),
                "information_gain": best_ig,
                "information_gain_by_target": best_ig_by_target,
                "p_gt_gain": best_p_gain,
                "p_gt_gain_by_target": best_p_by_target,
                "score_after": score_p_gt(current.p_gt, anchors, reduction=score_reduction),
                "p_gt_before": list(previous.p_gt),
                "p_gt_after": list(current.p_gt),
                "entropy_before": list(previous.entropy),
                "entropy_after": list(current.entropy),
            }
        )
    return step_rows, candidate_rows


def run_layout_control_probe(
    model,
    encoded: EncodedExample,
    *,
    max_k: int = 10,
    greedy_chain: list[int] | None = None,
    batch_size: int = 1,
    score_reduction: str = "mean",
) -> list[dict]:
    eligible = anchor_candidates(encoded)
    baseline = model.compute_p_gt_batch(encoded, [()], batch_size=batch_size)[0]

    specs: list[tuple[int, str, list[int]]] = []
    for k in range(1, min(max_k, len(eligible)) + 1):
        for name, anchors in layout_anchors(eligible, k, greedy_chain).items():
            specs.append((k, name, anchors))

    results = model.compute_p_gt_batch(encoded, [anchors for _, _, anchors in specs], batch_size=batch_size)

    rows: list[dict] = []
    for (k, name, anchors), after in zip(specs, results):
        ig_sum, ig_by_target = information_gain(baseline, after, exclude=anchors)
        p_gain_sum, p_gain_by_target = p_gt_gain(baseline, after, exclude=anchors)
        rows.append(
            {
                "example_id": encoded.example_id,
                "k": k,
                "layout": name,
                "anchors": list(anchors),
                "information_gain": ig_sum,
                "information_gain_by_target": ig_by_target,
                "p_gt_gain": p_gain_sum,
                "p_gt_gain_by_target": p_gain_by_target,
                "score_before": score_p_gt(baseline.p_gt, anchors, reduction=score_reduction),
                "score_after": score_p_gt(after.p_gt, anchors, reduction=score_reduction),
                "p_gt_default": list(baseline.p_gt),
                "p_gt_before": list(baseline.p_gt),
                "p_gt_after": list(after.p_gt),
                "entropy_before": list(baseline.entropy),
                "entropy_after": list(after.entropy),
            }
        )
    return rows


def anchor_candidates(encoded: EncodedExample) -> list[int]:
    if encoded.anchor_positions is not None:
        return list(encoded.anchor_positions)
    return list(range(encoded.completion_length))
