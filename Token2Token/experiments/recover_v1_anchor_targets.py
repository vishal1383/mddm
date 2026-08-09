#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from Token2Token.main.precompute_anchor_targets import source_fields
from Token2Token.main.train import encode_record, record_stream


def main() -> None:
    args = parse_args()
    log_path = Path(args.v1_log)
    config_path = Path(args.v1_config)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_args = SimpleNamespace(**config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], trust_remote_code=True
    )
    grouped = read_v1_targets(log_path)
    expected_examples = len(grouped)
    records = record_stream(train_args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        written = 0
        while written < expected_examples:
            source_record = next(records)
            encoded = encode_record(source_record, tokenizer, train_args)
            if encoded is None:
                continue
            prompt_ids, gold_ids, example_id = encoded
            rows = grouped.get(str(example_id))
            if rows is None:
                raise ValueError(f"missing V1 anchors for example {example_id}")
            targets = []
            for expected_rank, row in enumerate(rows, start=1):
                position = int(row["gold_position"])
                token_id = int(gold_ids[position])
                token_text = tokenizer.decode([token_id])
                if int(row["ig_rank"]) != expected_rank:
                    raise ValueError(f"non-consecutive ranks for example {example_id}")
                if token_text != str(row["anchor_token"]):
                    raise ValueError(
                        f"token recovery mismatch for example {example_id}, "
                        f"rank {expected_rank}: {token_text!r} != {row['anchor_token']!r}"
                    )
                targets.append(
                    {
                        "rank": expected_rank,
                        "gold_position": position,
                        "token_id": token_id,
                        "token": token_text,
                        "ig_score": float(row["ig_score"]),
                    }
                )
            recovered = {
                "dataset": config["dataset"],
                "example_id": str(example_id),
                "target_source": "v1_online_ig_recovered",
                "source": source_fields(config["dataset"], source_record[1]),
                "prompt_ids": prompt_ids,
                "gold_ids": gold_ids,
                "targets": targets,
            }
            handle.write(json.dumps(recovered, ensure_ascii=True) + "\n")
            written += 1
            if written % 100 == 0 or written == expected_examples:
                print(f"recovered={written}/{expected_examples}")
    temporary.replace(output)
    output.with_suffix(".config.json").write_text(
        json.dumps(
            {
                "target_source": "v1_online_ig_recovered",
                "v1_log": str(log_path),
                "v1_config": str(config_path),
                "examples": expected_examples,
                "anchors_per_example": sorted({len(rows) for rows in grouped.values()}),
                "model_id": config["model_id"],
                "dataset": config["dataset"],
                "seed": config["seed"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def read_v1_targets(path: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[str(row["example_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["ig_rank"]))
    counts = {len(rows) for rows in grouped.values()}
    if counts != {5}:
        raise ValueError(f"expected five V1 anchors per example, found {sorted(counts)}")
    return dict(grouped)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recover complete reusable anchor targets from the V1 log"
    )
    parser.add_argument("--v1-log", required=True)
    parser.add_argument("--v1-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
