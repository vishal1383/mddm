"""Pure-Python experiment contract and metric helpers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "gsm8k_temperature_passk_v2"
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
OFFICIAL_TEST_EXAMPLES = 1319
CANVAS_LENGTH = 256
BLOCK_LENGTH = 32
SAMPLES = 10
SEED = 42
PROMPT_SUFFIX = r" Please reason step by step, and put your final answer within \boxed{}."

METHODS = (
    "base",
    "jsd_mean_field",
    "dparallel",
    "paper_policy",
    "lora_sft",
    "dpo_policy_v3",
)
METHOD_LABELS = {
    "base": "Frozen Base confidence decoder",
    "jsd_mean_field": "JSD mean-field decoder",
    "dparallel": "dParallel",
    "paper_policy": "Unofficial Apple-method GRPO reproduction",
    "lora_sft": "Standard full-GSM8K LoRA SFT",
    "dpo_policy_v3": "Hidden-state select-then-sample DPO policy",
}
TEMPERATURES = (0.1, 0.5, 0.8, 1.2)


def task_matrix() -> tuple[tuple[str, float], ...]:
    return tuple((method, temperature) for method in METHODS for temperature in TEMPERATURES)


def task_for_id(task_id: int) -> tuple[str, float]:
    matrix = task_matrix()
    if not 0 <= int(task_id) < len(matrix):
        raise ValueError(f"task id must be in [0, {len(matrix) - 1}], got {task_id}")
    return matrix[int(task_id)]


def temperature_slug(temperature: float) -> str:
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    return f"T{float(temperature):.1f}"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def resume_compatible_contract(existing: dict[str, Any], current: dict[str, Any]) -> bool:
    """Allow partial legacy baselines to resume after DPO-only source edits."""

    if existing.get("method") == "dpo_policy_v3" or current.get("method") == "dpo_policy_v3":
        return False
    ignored = {"contract_sha256", "evaluator_sources_sha256"}
    old_semantics = {key: value for key, value in existing.items() if key not in ignored}
    new_semantics = {key: value for key, value in current.items() if key not in ignored}
    return old_semantics == new_semantics


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty run")
    paths = [path for record in records for path in record["paths"]]
    if any(len(record["paths"]) != SAMPLES for record in records):
        raise ValueError(f"every example must contain exactly {SAMPLES} paths")
    total_tokens = sum(int(path["generated_tokens"]) for path in paths)
    total_nfe = sum(int(path["base_forwards"]) for path in paths)
    if total_nfe <= 0:
        raise ValueError("total NFE must be positive")
    latency = sum(float(record["batch_latency_seconds"]) for record in records)
    correct = sum(bool(path["correct"]) for path in paths)
    return {
        "examples": len(records),
        "trajectories": len(paths),
        "sample_accuracy": correct / len(paths),
        "pass_at_5": sum(any(bool(path["correct"]) for path in record["paths"][:5]) for record in records)
        / len(records),
        "pass_at_10": sum(any(bool(path["correct"]) for path in record["paths"][:10]) for record in records)
        / len(records),
        "micro_tokens_per_nfe": total_tokens / total_nfe,
        "mean_nfe": total_nfe / len(paths),
        "total_nfe": total_nfe,
        "total_generated_tokens": total_tokens,
        "synchronized_latency_seconds": latency,
        "end_to_end_tokens_per_second": total_tokens / latency if latency > 0 else None,
        "mean_unique_normalized_answers_at_10": sum(int(record["unique_normalized_answers_at_10"]) for record in records)
        / len(records),
        "mean_unique_traces_at_10": sum(int(record["unique_traces_at_10"]) for record in records)
        / len(records),
    }


def iter_json_files(directory: str | Path) -> Iterable[Path]:
    yield from sorted(Path(directory).glob("*.json"))
