from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

from aggregate import collect
from experiment_contract import (
    CANVAS_LENGTH,
    OFFICIAL_TEST_EXAMPLES,
    SCHEMA_VERSION,
    METHODS,
    SAMPLES,
    TEMPERATURES,
    summarize_records,
    task_for_id,
    task_matrix,
    temperature_slug,
)


class ContractTest(unittest.TestCase):
    def test_task_matrix_is_complete_and_stable(self) -> None:
        matrix = task_matrix()
        self.assertEqual(len(matrix), 72)
        self.assertEqual(len(set(matrix)), 72)
        self.assertEqual(task_for_id(0), (METHODS[0], TEMPERATURES[0]))
        self.assertEqual(task_for_id(71), (METHODS[-1], TEMPERATURES[-1]))
        self.assertEqual(temperature_slug(1.2), "T1.2")

    def test_pass_metrics_and_micro_throughput(self) -> None:
        records = []
        for example in range(2):
            paths = []
            for index in range(SAMPLES):
                paths.append(
                    {
                        "correct": (example == 0 and index == 7) or (example == 1 and index == 2),
                        "generated_tokens": CANVAS_LENGTH,
                        "base_forwards": 32,
                    }
                )
            records.append(
                {
                    "paths": paths,
                    "batch_latency_seconds": 2.0,
                    "unique_normalized_answers_at_10": 4,
                    "unique_traces_at_10": 8,
                }
            )
        summary = summarize_records(records)
        self.assertAlmostEqual(summary["sample_accuracy"], 0.1)
        self.assertAlmostEqual(summary["pass_at_5"], 0.5)
        self.assertAlmostEqual(summary["pass_at_10"], 1.0)
        self.assertAlmostEqual(summary["micro_tokens_per_nfe"], 8.0)
        self.assertAlmostEqual(summary["end_to_end_tokens_per_second"], 1280.0)

    def test_aggregator_requires_and_accepts_all_seventy_two_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                collect(root)
            for method, temperature in task_matrix():
                cell = root / method / temperature_slug(temperature)
                path = cell / "summary.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                contract = {
                    "schema": SCHEMA_VERSION,
                    "method": method,
                    "temperature": temperature,
                    "temperature_semantics": "categorical_softmax_logits_div_T",
                    "dataset": "openai/gsm8k:main:test",
                    "start": 0,
                    "stop": OFFICIAL_TEST_EXAMPLES,
                    "samples": 10,
                    "canvas_length": 256,
                    "block_length": 32,
                    "prompt_suffix": "same",
                    "seed": 42,
                    "one_base_forward_per_cycle": True,
                    "contract_sha256": f"receipt-{method}-{temperature}",
                }
                (cell / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
                path.write_text(
                    json.dumps(
                        {
                            "schema": SCHEMA_VERSION,
                            "method": method,
                            "temperature": temperature,
                            "examples": OFFICIAL_TEST_EXAMPLES,
                            "trajectories": OFFICIAL_TEST_EXAMPLES * 10,
                            "complete": True,
                            "contract_sha256": contract["contract_sha256"],
                        }
                    ),
                    encoding="utf-8",
                )
            self.assertEqual(len(collect(root)), 72)


if __name__ == "__main__":
    unittest.main()
