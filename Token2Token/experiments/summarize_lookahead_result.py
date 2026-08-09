#!/usr/bin/env python3
"""Build a paired quality and latency report from saved decoder generations."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from Token2Token.main.paired_comparison import compare, load_rows


def main() -> None:
    args = parse_args()
    baseline = load_rows(Path(args.baseline_predictions))
    trained = load_rows(Path(args.trained_predictions))
    shared = sorted(set(baseline) & set(trained), key=int)
    if not shared:
        raise SystemExit("no shared prediction rows")

    base_summary = load_summary(Path(args.baseline_summary))
    trained_summary = load_summary(Path(args.trained_summary))
    reference_summaries = [
        load_summary(Path(path)) for path in args.reference_summary
    ]
    all_result = summarize_slice(
        baseline,
        trained,
        shared,
        args.completion_length,
        args.batch_size,
    )
    untouched_ids = [
        key
        for key in shared
        if args.untouched_start <= int(key) < args.untouched_end
    ]
    untouched_result = summarize_slice(
        baseline,
        trained,
        untouched_ids,
        args.completion_length,
        args.batch_size,
    )
    gate_rows = load_gate_rows(Path(args.gate_root)) if args.gate_root else []

    report = render_report(
        all_result,
        untouched_result,
        base_summary,
        trained_summary,
        gate_rows,
        reference_summaries,
        args,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(
            {
                "all": all_result,
                "untouched": untouched_result,
                "baseline_summary": base_summary,
                "trained_summary": trained_summary,
                "reference_summaries": reference_summaries,
                "gate": gate_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(report)


def summarize_slice(baseline, trained, ids, completion_length, batch_size):
    base_rows = [baseline[key] for key in ids]
    trained_rows = [trained[key] for key in ids]
    paired = compare(base_rows, trained_rows)
    deltas = [
        int(b["correct"]) - int(a["correct"])
        for a, b in zip(base_rows, trained_rows)
    ]
    return {
        "examples": len(ids),
        "min_example_id": min(map(int, ids)),
        "max_example_id": max(map(int, ids)),
        "accuracy_delta_ci95": bootstrap_mean_ci(deltas),
        "paired": paired,
        "baseline": summarize_arm(base_rows, completion_length, batch_size),
        "trained": summarize_arm(trained_rows, completion_length, batch_size),
    }


def summarize_arm(rows, completion_length, batch_size):
    examples = len(rows)
    total_tokens = completion_length * examples
    total_forwards = sum(int(row["model_forwards"]) for row in rows)
    batches = defaultdict(list)
    for row in rows:
        batches[int(row["example_id"]) // batch_size].append(row)
    batch_iterations = sum(
        max(int(row["model_forwards"]) for row in batch) for batch in batches.values()
    )
    computed_row_forwards = sum(
        len(batch) * max(int(row["model_forwards"]) for row in batch)
        for batch in batches.values()
    )
    cycles = sum(int(row["cycles"]) for row in rows)
    catalyst = sum(int(row["catalyst_tokens"]) for row in rows)
    cleanup = sum(int(row["cleanup_tokens"]) for row in rows)
    threshold = sum(int(row["threshold_tokens"]) for row in rows)
    primary_catalysts = cycles - cleanup
    second_catalysts = catalyst - primary_catalysts
    if catalyst + cleanup + threshold != total_tokens:
        raise ValueError("commit phases do not sum to the completion canvas")
    if second_catalysts < 0:
        raise ValueError("invalid catalyst accounting")
    return {
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / examples,
        "total_model_forwards": total_forwards,
        "forwards_per_example": total_forwards / examples,
        "tokens_per_forward_batch1": total_tokens / total_forwards,
        "batch_iterations": batch_iterations,
        "computed_row_forwards": computed_row_forwards,
        "tokens_per_computed_row_forward": total_tokens / computed_row_forwards,
        "batch_padding_overhead": computed_row_forwards / total_forwards - 1,
        "cycles": cycles,
        "catalyst_tokens": catalyst,
        "primary_catalyst_tokens": primary_catalysts,
        "second_catalyst_tokens": second_catalysts,
        "cleanup_tokens": cleanup,
        "threshold_tokens": threshold,
        "tokens_per_cycle": total_tokens / cycles,
        "threshold_tokens_per_cycle": threshold / cycles,
        "second_catalyst_rate": second_catalysts / primary_catalysts,
    }


def bootstrap_mean_ci(values, samples=10_000, seed=0):
    rng = random.Random(seed)
    count = len(values)
    means = sorted(sum(rng.choices(values, k=count)) / count for _ in range(samples))
    return [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]


def load_summary(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"expected exactly one summary in {path}")
    return value[0]


def load_gate_rows(root):
    rows = []
    for path in sorted(root.glob("*merged*/summary.json")):
        summary = load_summary(path)
        rows.append(
            {
                "name": path.parent.name,
                "correct": summary["correct"],
                "examples": summary["examples"],
                "forwards": summary["total_model_forwards"],
                "tokens_per_forward": summary["tokens_per_forward"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
    return rows


def render_report(
    all_result,
    untouched,
    base_summary,
    trained_summary,
    gate,
    references,
    args,
):
    all_base = all_result["baseline"]
    all_trained = all_result["trained"]
    wall_base = float(base_summary["elapsed_seconds"])
    wall_trained = float(trained_summary["elapsed_seconds"])
    total_tokens = args.completion_length * all_result["examples"]
    lines = [
        "# Threshold-Lookahead Final Report",
        "",
        "LLaDA-8B-Instruct on GSM8K test. Both arms use the same adaptive "
        "decoder: text tau=.90, numeric tau=.99, and at most two jointly "
        "selected catalysts with p2>=.60 and p2/p1>=.85.",
        "",
        "## Main result",
        "",
        "| Slice | Arm | Correct | Accuracy | Forwards/example | Tokens/forward (batch-1) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    append_slice_rows(lines, "All test", all_result)
    append_slice_rows(lines, f"Untouched {args.untouched_start}-{args.untouched_end - 1}", untouched)
    paired = all_result["paired"]
    ci_low, ci_high = all_result["accuracy_delta_ci95"]
    untouched_paired = untouched["paired"]
    untouched_low, untouched_high = untouched["accuracy_delta_ci95"]
    lines.extend(
        [
            "",
            f"All-test paired churn: {paired['only_baseline_correct']} base-only "
            f"correct, {paired['only_trained_correct']} trained-only correct; "
            f"exact McNemar p={paired['mcnemar_p_value']:.4f}. The paired accuracy "
            f"delta is {100 * (all_trained['accuracy'] - all_base['accuracy']):+.2f} pp "
            f"(bootstrap 95% CI {100 * ci_low:+.2f} to {100 * ci_high:+.2f} pp).",
            f"Untouched paired churn: {untouched_paired['only_baseline_correct']} "
            f"base-only correct, {untouched_paired['only_trained_correct']} "
            f"trained-only correct; exact McNemar "
            f"p={untouched_paired['mcnemar_p_value']:.4f}. Its accuracy delta "
            f"95% CI is {100 * untouched_low:+.2f} to "
            f"{100 * untouched_high:+.2f} pp.",
            "",
            "## Latency",
            "",
            "| Arm | Summed per-example forwards | Batch-16 iterations | Computed row-forwards | Wall time | Canvas tokens/s |",
            "|---|---:|---:|---:|---:|---:|",
            latency_row("Base", all_base, wall_base, total_tokens),
            latency_row("Trained", all_trained, wall_trained, total_tokens),
            "",
            f"Forward reduction (batch-1): {100 * (1 - all_trained['total_model_forwards'] / all_base['total_model_forwards']):.1f}%. "
            f"Batch-16 iteration reduction: {100 * (1 - all_trained['batch_iterations'] / all_base['batch_iterations']):.1f}%. "
            f"Measured wall-time reduction: {100 * (1 - wall_trained / wall_base):.1f}%.",
            "",
            "## Commit mechanism",
            "",
            "| Arm | Tokens/cycle | Threshold tokens/cycle | Second catalysts | Second-catalyst rate | Cleanup tokens |",
            "|---|---:|---:|---:|---:|---:|",
            mechanism_row("Base", all_base),
            mechanism_row("Trained", all_trained),
            "",
            f"Threshold bursts explain {100 * ((all_trained['threshold_tokens_per_cycle'] - all_base['threshold_tokens_per_cycle']) / (all_trained['tokens_per_cycle'] - all_base['tokens_per_cycle'])):.1f}% "
            "of the tokens-per-cycle increase. The adapter also raises second-"
            f"catalyst acceptance from {100 * all_base['second_catalyst_rate']:.1f}% "
            f"to {100 * all_trained['second_catalyst_rate']:.1f}%.",
            "",
            "## Standard decoder context",
            "",
            "| Model | Correct | Accuracy | Tokens/forward | Wall time |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for summary in references:
        lines.append(
            f"| {summary['model_label']} | {summary['correct']}/{summary['examples']} | "
            f"{100 * summary['accuracy']:.2f}% | {summary['tokens_per_forward']:.3f} | "
            f"{summary['elapsed_seconds'] / 60:.1f} min |"
        )
    lines.extend(
        [
            f"| Matched adaptive base | {all_base['correct']}/{all_result['examples']} | "
            f"{100 * all_base['accuracy']:.2f}% | {all_base['tokens_per_forward_batch1']:.3f} | "
            f"{wall_base / 60:.1f} min |",
            f"| Threshold-lookahead trained | {all_trained['correct']}/{all_result['examples']} | "
            f"{100 * all_trained['accuracy']:.2f}% | {all_trained['tokens_per_forward_batch1']:.3f} | "
            f"{wall_trained / 60:.1f} min |",
            "",
            "The trained adaptive model Pareto-improves its matched adaptive "
            "base. It does not yet dominate standard block decoding: compared "
            "with base block-32 k=2 it is faster but lower-accuracy.",
            "",
            "## Merged checkpoint gate (test IDs 128-191)",
            "",
            "| Checkpoint | Correct | Forwards | Tokens/forward | Wall time |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in gate:
        lines.append(
            f"| {row['name']} | {row['correct']}/{row['examples']} | "
            f"{row['forwards']:,} | {row['tokens_per_forward']:.3f} | "
            f"{row['elapsed_seconds']:.1f}s |"
        )
    lines.extend(
        [
            "",
            "Checkpoint 6000 was promoted because it tied for the best merged "
            "accuracy and used the fewest forwards. The ordering is a selection "
            "heuristic on 64 examples, not evidence that checkpoint 6000 is "
            "intrinsically better than the tied checkpoints.",
            "",
            "## Scope",
            "",
            "The adapter was trained only on the GSM8K train split. Decoder "
            "thresholds and checkpoint selection touched test IDs 0-191, so the "
            "192-1318 result is the primary untouched estimate. Forward metrics "
            "describe logical batch-1 efficiency; measured batch-16 wall time is "
            "reported separately because variable sequence completion causes "
            "padding waste.",
            "",
        ]
    )
    return "\n".join(lines)


def append_slice_rows(lines, label, result):
    for arm_label, key in (("Base", "baseline"), ("Trained", "trained")):
        arm = result[key]
        lines.append(
            f"| {label} | {arm_label} | {arm['correct']}/{result['examples']} | "
            f"{100 * arm['accuracy']:.2f}% | {arm['forwards_per_example']:.2f} | "
            f"{arm['tokens_per_forward_batch1']:.3f} |"
        )


def latency_row(label, arm, elapsed, total_tokens):
    return (
        f"| {label} | {arm['total_model_forwards']:,} | "
        f"{arm['batch_iterations']:,} | {arm['computed_row_forwards']:,} | "
        f"{elapsed / 60:.1f} min | {total_tokens / elapsed:.2f} |"
    )


def mechanism_row(label, arm):
    return (
        f"| {label} | {arm['tokens_per_cycle']:.3f} | "
        f"{arm['threshold_tokens_per_cycle']:.3f} | "
        f"{arm['second_catalyst_tokens']:,} | "
        f"{100 * arm['second_catalyst_rate']:.1f}% | "
        f"{arm['cleanup_tokens']:,} |"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--trained-predictions", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--trained-summary", required=True)
    parser.add_argument("--gate-root")
    parser.add_argument("--reference-summary", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--completion-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--untouched-start", type=int, default=192)
    parser.add_argument("--untouched-end", type=int, default=1319)
    return parser.parse_args()


if __name__ == "__main__":
    main()
