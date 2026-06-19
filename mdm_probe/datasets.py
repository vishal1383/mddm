from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .types import ProbeExample


def load_examples(
    dataset: str,
    *,
    split: str = "test",
    limit: int | None = None,
    data_path: str | None = None,
    prompt_field: str = "prompt",
    completion_field: str = "completion",
) -> list[ProbeExample]:
    """Load prompt/completion examples for probing."""

    if data_path:
        examples = list(_load_local_file(data_path, dataset, prompt_field, completion_field))
    elif dataset == "gsm8k":
        examples = list(_load_gsm8k(split))
    elif dataset == "humaneval":
        examples = list(_load_humaneval(split))
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Use gsm8k, humaneval, or pass --data-path."
        )

    if limit is not None:
        examples = examples[:limit]
    return examples


def _load_gsm8k(split: str) -> Iterable[ProbeExample]:
    load_dataset = _require_datasets()
    try:
        rows = load_dataset("openai/gsm8k", "main", split=split)
    except Exception:
        rows = load_dataset("gsm8k", "main", split=split)
    for idx, row in enumerate(rows):
        yield ProbeExample(
            example_id=str(row.get("id", idx)),
            dataset="gsm8k",
            prompt=str(row["question"]).rstrip() + "\n",
            completion=str(row["answer"]).strip(),
            metadata={"source_index": idx},
        )


def _load_humaneval(split: str) -> Iterable[ProbeExample]:
    load_dataset = _require_datasets()
    try:
        rows = load_dataset("openai/openai_humaneval", split=split)
    except Exception:
        rows = load_dataset("openai_humaneval", split=split)
    for idx, row in enumerate(rows):
        task_id = str(row.get("task_id", idx))
        yield ProbeExample(
            example_id=task_id,
            dataset="humaneval",
            prompt=str(row["prompt"]),
            completion=str(row["canonical_solution"]).rstrip(),
            metadata={"task_id": task_id, "source_index": idx},
        )


def _load_local_file(
    data_path: str,
    dataset: str,
    prompt_field: str,
    completion_field: str,
) -> Iterable[ProbeExample]:
    path = Path(data_path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                yield _row_to_example(row, idx, dataset, prompt_field, completion_field)
    elif path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        for idx, row in enumerate(rows):
            yield _row_to_example(row, idx, dataset, prompt_field, completion_field)
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            for idx, row in enumerate(csv.DictReader(handle)):
                yield _row_to_example(row, idx, dataset, prompt_field, completion_field)
    else:
        raise ValueError("Local data must be .jsonl, .json, or .csv")


def _row_to_example(
    row: dict,
    idx: int,
    dataset: str,
    prompt_field: str,
    completion_field: str,
) -> ProbeExample:
    return ProbeExample(
        example_id=str(row.get("id", row.get("task_id", idx))),
        dataset=dataset,
        prompt=str(row[prompt_field]),
        completion=str(row[completion_field]),
        metadata={k: v for k, v in row.items() if k not in {prompt_field, completion_field}},
    )


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the 'datasets' package to load GSM8K/HumanEval.") from exc
    return load_dataset
