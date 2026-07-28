#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mdm_probe.datasets import load_examples
from mdm_probe.models import load_probe_model
from mdm_probe.plots import plot_greedy_results, plot_layout_results, plot_single_anchor_results
from mdm_probe.sampling import (
    run_greedy_k_probe,
    run_layout_control_probe,
    run_single_anchor_probe,
)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_probe_model(
        args.model,
        model_id=args.model_id,
        device=args.device,
        mask_token_id=args.mask_token_id,
        mask_token=args.mask_token,
        prompt_format=args.prompt_format,
    )
    examples = load_examples(
        args.dataset,
        split=args.split,
        limit=args.limit,
        data_path=args.data_path,
        prompt_field=args.prompt_field,
        completion_field=args.completion_field,
    )

    single_rows: list[dict] = []
    greedy_rows: list[dict] = []
    greedy_candidate_rows: list[dict] = []
    layout_rows: list[dict] = []
    example_rows: list[dict] = []

    for index, example in enumerate(examples):
        encoded = model.encode_example(
            example,
            max_completion_tokens=args.max_completion_tokens,
        )
        example_rows.append(
            {
                "example_id": encoded.example_id,
                "dataset": example.dataset,
                "prompt_tokens": encoded.prompt_length,
                "completion_tokens": encoded.completion_length,
                "prompt": encoded.prompt_text,
                "completion": encoded.completion_text,
                "completion_token_ids": encoded.completion_token_ids,
                "completion_token_texts": encoded.completion_token_texts,
                "anchor_positions": encoded.anchor_positions,
                "prompt_format": model.prompt_format,
            }
        )
        print(f"[{index + 1}/{len(examples)}] {encoded.example_id}: T={encoded.completion_length}")

        greedy_chain: list[int] | None = None

        if args.probe in {"single", "all"}:
            rows = run_single_anchor_probe(
                model,
                encoded,
                batch_size=args.batch_size,
                score_reduction=args.score_reduction,
                selection_metric=args.select_metric,
            )
            single_rows.extend(rows)

        if args.probe in {"greedy", "layout", "all"}:
            greedy_name = greedy_layout_name(args.select_metric)
            rows, candidate_rows = run_greedy_k_probe(
                model,
                encoded,
                max_k=args.max_k,
                batch_size=args.batch_size,
                score_reduction=args.score_reduction,
                selection_metric=args.select_metric,
            )
            greedy_chain = [row["selected_anchor_position"] for row in rows]
            if args.probe in {"greedy", "all"}:
                greedy_rows.extend(rows)
                greedy_candidate_rows.extend(candidate_rows)

        if args.probe in {"layout", "all"}:
            rows = run_layout_control_probe(
                model,
                encoded,
                max_k=args.max_k,
                greedy_chain=greedy_chain,
                greedy_name=greedy_name,
                batch_size=args.batch_size,
                score_reduction=args.score_reduction,
            )
            layout_rows.extend(rows)

    write_rows(out_dir / "examples.jsonl", example_rows)
    write_rows(out_dir / "examples.csv", example_rows)

    if single_rows:
        write_rows(out_dir / "single_anchor.jsonl", single_rows)
        write_rows(out_dir / "single_anchor.csv", single_rows)
        single_best = best_rows(single_rows, "example_id", args.select_metric)
        write_rows(out_dir / "single_anchor_best_by_prompt.jsonl", single_best)
        write_rows(out_dir / "single_anchor_best_by_prompt.csv", single_best)
        write_rows(out_dir / "single_anchor_aggregate.jsonl", aggregate_rows(single_best, []))
        write_rows(out_dir / "single_anchor_aggregate.csv", aggregate_rows(single_best, []))
        if not args.no_plots:
            plot_single_anchor_results(single_rows, plots_dir)

    if greedy_rows:
        write_rows(out_dir / "greedy_k.jsonl", greedy_rows)
        write_rows(out_dir / "greedy_k.csv", greedy_rows)
        write_rows(out_dir / "greedy_candidates.jsonl", greedy_candidate_rows)
        write_rows(out_dir / "greedy_candidates.csv", greedy_candidate_rows)
        write_rows(out_dir / "greedy_k_aggregate.jsonl", aggregate_rows(greedy_rows, ["k"]))
        write_rows(out_dir / "greedy_k_aggregate.csv", aggregate_rows(greedy_rows, ["k"]))
        if not args.no_plots:
            plot_greedy_results(greedy_rows, plots_dir)

    if layout_rows:
        write_rows(out_dir / "layout_control.jsonl", layout_rows)
        write_rows(out_dir / "layout_control.csv", layout_rows)
        write_rows(out_dir / "layout_control_aggregate.jsonl", aggregate_rows(layout_rows, ["layout", "k"]))
        write_rows(out_dir / "layout_control_aggregate.csv", aggregate_rows(layout_rows, ["layout", "k"]))
        if not args.no_plots:
            plot_layout_results(layout_rows, plots_dir)

    print(f"Saved outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MDM spread-anchor probes.")
    parser.add_argument("--model", required=True, help="llada-8b, dream-7b, or HF model path")
    parser.add_argument("--model-id", default=None, help="Optional HF model id/path override")
    parser.add_argument("--dataset", default="gsm8k", help="gsm8k, humaneval, or local dataset name")
    parser.add_argument("--data-path", default=None, help="Optional local .jsonl/.json/.csv")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--completion-field", default="completion")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--probe", choices=["single", "greedy", "layout", "all"], default="all")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--score-reduction", choices=["sum", "mean"], default="mean")
    parser.add_argument(
        "--select-metric",
        choices=["information_gain", "p_gt_gain", "max_p_gain"],
        default="information_gain",
        help="Metric used to pick greedy anchors; default preserves the original entropy-IG behavior.",
    )
    parser.add_argument("--out-dir", default="outputs/mdm_probe")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--mask-token", default=None)
    parser.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def best_rows(rows: list[dict], group_key: str, score_key: str) -> list[dict]:
    best: dict[Any, dict] = {}
    for row in rows:
        key = row[group_key]
        if key not in best or row[score_key] > best[key][score_key]:
            best[key] = row
    return list(best.values())


def aggregate_rows(rows: list[dict], group_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    out: list[dict] = []
    for key, group in sorted(groups.items()):
        aggregate = {k: v for k, v in zip(group_keys, key)}
        aggregate["n"] = len(group)
        for metric in [
            "information_gain",
            "p_gt_gain",
            "max_p_gain",
            "score_before",
            "score_after",
            "max_p_score_before",
            "max_p_score_after",
        ]:
            values = [row[metric] for row in group if isinstance(row.get(metric), (int, float))]
            if values:
                aggregate[f"mean_{metric}"] = sum(values) / len(values)
        out.append(aggregate)
    return out


def greedy_layout_name(select_metric: str) -> str:
    return {
        "information_gain": "greedy_ig",
        "p_gt_gain": "greedy_p_gt",
        "max_p_gain": "greedy_max_p",
    }[select_metric]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


if __name__ == "__main__":
    main()
