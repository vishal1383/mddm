#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def summarize_epochs(
    rows: list[dict], steps_per_epoch: int, run_dir: Path, step_offset: int = 0
):
    complete_epochs = len(rows) // steps_per_epoch
    summaries = []
    for index in range(complete_epochs):
        start = index * steps_per_epoch
        end = start + steps_per_epoch
        epoch_rows = rows[start:end]
        checkpoint_step = step_offset + end
        summaries.append(
            {
                "epoch": step_offset // steps_per_epoch + index + 1,
                "checkpoint_step": checkpoint_step,
                "adapter_path": str(run_dir / f"checkpoint-{checkpoint_step:06d}"),
                "mean_loss": statistics.fmean(row["loss"] for row in epoch_rows),
                "mean_anchor_loss": statistics.fmean(
                    row["anchor_loss"] for row in epoch_rows
                ),
                "mean_sequence_loss": statistics.fmean(
                    row["sequence_loss"] for row in epoch_rows
                ),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    train_log = Path(args.train_log)
    with train_log.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    summaries = summarize_epochs(
        rows, args.steps_per_epoch, Path(args.run_dir), args.step_offset
    )
    if not summaries:
        raise ValueError("training log contains no complete epoch")
    best = min(summaries, key=lambda row: row["mean_loss"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"epochs": summaries, "best": best}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(best["adapter_path"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select the completed epoch with minimum mean training loss"
    )
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps-per-epoch", type=int, required=True)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.steps_per_epoch <= 0:
        parser.error("steps-per-epoch must be positive")
    if args.step_offset < 0 or args.step_offset % args.steps_per_epoch:
        parser.error("step-offset must be a non-negative epoch boundary")
    return args


if __name__ == "__main__":
    main()
