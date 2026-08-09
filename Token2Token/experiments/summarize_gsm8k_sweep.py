#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_LABELS = {
    "anchor_lora": "IG-anchor LoRA",
    "standard_lora": "Standard LoRA",
    "base": "Base LLaDA",
}


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summaries = {
        model: read_summary(root / model / "summary.json")
        for model in MODEL_LABELS
    }
    k_values = sorted(parse_k_values(args.k_values))
    rows = build_rows(summaries, k_values)
    report = render_report(rows)
    (root / "final_results.md").write_text(report, encoding="utf-8")
    (root / "final_results.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(report)


def read_summary(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["tokens_per_step"]): item for item in items}


def build_rows(summaries: dict[str, dict[int, dict]], k_values) -> list[dict]:
    rows = []
    for k in k_values:
        row = {"tokens_per_step": k}
        for model in MODEL_LABELS:
            item = summaries.get(model, {}).get(k)
            row[model] = result_fields(item)
        anchor = row["anchor_lora"]["accuracy"]
        standard = row["standard_lora"]["accuracy"]
        base = row["base"]["accuracy"]
        row["anchor_minus_standard_pp"] = percentage_point_delta(anchor, standard)
        row["anchor_minus_base_pp"] = percentage_point_delta(anchor, base)
        rows.append(row)
    return rows


def result_fields(item: dict | None) -> dict:
    if item is None:
        return {"accuracy": None, "correct": 0, "examples": 0}
    return {
        "accuracy": float(item["accuracy"]),
        "correct": int(item["correct"]),
        "examples": int(item["examples"]),
    }


def percentage_point_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return 100.0 * (left - right)


def render_report(rows: list[dict]) -> str:
    lines = [
        "# GSM8K Top-k Confidence Decode Results",
        "",
        "All models use LLaDA-8B-Instruct with a 128-token completion canvas. "
        "At each decode step, the model commits the top-k most confident masked "
        "positions.",
        "",
        "| k | IG-anchor LoRA | Standard LoRA | Base LLaDA | Anchor - Standard | Anchor - Base |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {k} | {anchor} | {standard} | {base} | {d_standard} | {d_base} |".format(
                k=row["tokens_per_step"],
                anchor=format_result(row["anchor_lora"]),
                standard=format_result(row["standard_lora"]),
                base=format_result(row["base"]),
                d_standard=format_delta(row["anchor_minus_standard_pp"]),
                d_base=format_delta(row["anchor_minus_base_pp"]),
            )
        )
    lines.extend(
        [
            "",
            "Standard LoRA is the normal masked-denoising fine-tuning control. "
            "It does not use IG anchors, Gaussian placement, or relative-order loss.",
            "",
        ]
    )
    return "\n".join(lines)


def format_result(result: dict) -> str:
    if result["accuracy"] is None:
        return "pending"
    return (
        f"{100.0 * result['accuracy']:.2f}% "
        f"({result['correct']}/{result['examples']})"
    )


def format_delta(value: float | None) -> str:
    return "pending" if value is None else f"{value:+.2f} pp"


def parse_k_values(value: str) -> list[int]:
    values = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not values or any(item <= 0 for item in values):
        raise ValueError("k-values must contain positive integers")
    return list(values)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize the GSM8K k sweep")
    parser.add_argument("--root", required=True)
    parser.add_argument("--k-values", default="5,4,3,2,1")
    return parser.parse_args()


if __name__ == "__main__":
    main()
