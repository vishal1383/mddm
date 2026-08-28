#!/usr/bin/env python3
"""Validate completed cells and render the matched CSV/Markdown tables."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

from experiment_contract import (
    METHOD_LABELS,
    OFFICIAL_TEST_EXAMPLES,
    SCHEMA_VERSION,
    task_matrix,
    temperature_slug,
)


FIELDS = (
    "method",
    "temperature",
    "examples",
    "sample_accuracy",
    "pass_at_5",
    "pass_at_10",
    "micro_tokens_per_nfe",
    "mean_nfe",
    "total_nfe",
    "synchronized_latency_seconds",
    "end_to_end_tokens_per_second",
    "mean_unique_normalized_answers_at_10",
    "mean_unique_traces_at_10",
)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def collect(output_root: Path, allow_partial: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    shared_contract: dict[str, Any] | None = None
    shared_fields = (
        "schema",
        "temperature_semantics",
        "dataset",
        "start",
        "stop",
        "samples",
        "canvas_length",
        "block_length",
        "prompt_suffix",
        "seed",
        "one_base_forward_per_cycle",
    )
    for method, temperature in task_matrix():
        path = output_root / method / temperature_slug(temperature) / "summary.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        contract_path = path.with_name("contract.json")
        if not contract_path.is_file():
            raise FileNotFoundError(f"summary has no matching contract: {contract_path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        errors = []
        if summary.get("schema") != SCHEMA_VERSION:
            errors.append("schema")
        if summary.get("method") != method or float(summary.get("temperature", -1)) != temperature:
            errors.append("cell identity")
        if int(summary.get("examples", -1)) != OFFICIAL_TEST_EXAMPLES or not summary.get("complete"):
            errors.append("full-dataset completion")
        if int(summary.get("trajectories", -1)) != OFFICIAL_TEST_EXAMPLES * 10:
            errors.append("ten trajectories per example")
        if summary.get("contract_sha256") != contract.get("contract_sha256"):
            errors.append("summary/contract hash")
        if contract.get("method") != method or float(contract.get("temperature", -1)) != temperature:
            errors.append("contract identity")
        if shared_contract is None:
            shared_contract = contract
        elif any(contract.get(field) != shared_contract.get(field) for field in shared_fields):
            errors.append("shared matched contract")
        if errors:
            raise ValueError(f"invalid summary {path}: {', '.join(errors)}")
        rows.append(summary)
    if missing and not allow_partial:
        preview = "\n".join(missing[:8])
        raise FileNotFoundError(f"{len(missing)} of {len(task_matrix())} result cells are missing:\n{preview}")
    return rows


def csv_text(rows: Sequence[dict[str, Any]]) -> str:
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def markdown_text(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# GSM8K temperature/pass@k results",
        "",
        "| Method | T | Sample acc. | pass@5 | pass@10 | Tok/NFE | Mean NFE | Tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    previous_method = None
    for row in rows:
        tokens_second = row.get("end_to_end_tokens_per_second")
        method_label = METHOD_LABELS[row["method"]] if row["method"] != previous_method else ""
        lines.append(
            "| {label} | {temperature:.1f} | {accuracy:.2%} | {pass5:.2%} | {pass10:.2%} | "
            "{tok_nfe:.3f} | {mean_nfe:.2f} | {tok_s} |".format(
                label=method_label,
                temperature=float(row["temperature"]),
                accuracy=float(row["sample_accuracy"]),
                pass5=float(row["pass_at_5"]),
                pass10=float(row["pass_at_10"]),
                tok_nfe=float(row["micro_tokens_per_nfe"]),
                mean_nfe=float(row["mean_nfe"]),
                tok_s=f"{float(tokens_second):.2f}" if tokens_second is not None else "n/a",
            )
        )
        previous_method = row["method"]
    lines.extend(
        [
            "",
            "Sample accuracy is marginal accuracy over all ten trajectories. pass@5 uses paths 0–4; pass@10 uses paths 0–9.",
            "Tok/NFE is the micro ratio over all generated tokens and all full model forwards.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--table-stem", choices=("final_table", "baseline_60_table"), default="final_table")
    args = parser.parse_args(argv)
    root = Path(args.output_root).resolve()
    rows = collect(root, allow_partial=args.allow_partial)
    rows.sort(key=lambda row: (list(METHOD_LABELS).index(row["method"]), float(row["temperature"])))
    table_dir = root / "tables"
    atomic_text(table_dir / f"{args.table_stem}.csv", csv_text(rows))
    atomic_text(table_dir / f"{args.table_stem}.md", markdown_text(rows))
    summary_name = "all_summaries.json" if args.table_stem == "final_table" else "baseline_60_all_summaries.json"
    atomic_text(table_dir / summary_name, json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(markdown_text(rows))


if __name__ == "__main__":
    main()
