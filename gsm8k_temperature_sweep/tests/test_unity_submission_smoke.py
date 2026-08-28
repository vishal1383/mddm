from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class UnitySubmissionSmokeTest(unittest.TestCase):
    def test_real_launcher_builds_the_complete_sequential_dependency_chain(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_bin = temporary / "bin"
            fake_venv = temporary / "venv/bin"
            fake_bin.mkdir(parents=True)
            fake_venv.mkdir(parents=True)
            fake_python = fake_venv / "python"
            fake_python.write_text(
                """#!/usr/bin/env bash
if [[ "${1:-}" == *seal_model_revisions.py ]]; then
  echo base-revision-sha dparallel-revision-sha
fi
exit 0
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            calls_path = temporary / "sbatch_calls.jsonl"
            counter_path = temporary / "sbatch_counter"
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                """#!/usr/bin/env python3
import json, os, pathlib, sys
counter = pathlib.Path(os.environ["FAKE_SBATCH_COUNTER"])
value = int(counter.read_text()) if counter.exists() else 4100
counter.write_text(str(value + 1))
with pathlib.Path(os.environ["FAKE_SBATCH_CALLS"]).open("a") as handle:
    handle.write(json.dumps({"id": value, "argv": sys.argv[1:], "dpo": os.environ.get("DPO_POLICY_CHECKPOINT"), "base": os.environ.get("BASE_MODEL_REVISION"), "dparallel": os.environ.get("DPARALLEL_MODEL_REVISION")}) + "\\n")
print(value)
""",
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o755)
            output_root = temporary / "results"
            env = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "MDDM_SWEEP_VENV": str(temporary / "venv"),
                "ML_RL_DLLM_REPO": str(temporary / "policy-repo"),
                "MDDM_SWEEP_OUTPUT_ROOT": str(output_root),
                "FAKE_SBATCH_CALLS": str(calls_path),
                "FAKE_SBATCH_COUNTER": str(counter_path),
            }
            completed = subprocess.run(
                ["bash", str(root / "scripts/submit.sh")],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([call["id"] for call in calls], [4100, 4101, 4102, 4103, 4104])
        self.assertIn("--array=0-59%1", calls[0]["argv"])
        self.assertIn("--gpus=a100:1", calls[0]["argv"])
        self.assertIn("--dependency=afterok:4100", calls[1]["argv"])
        self.assertIn("--export=ALL,MDDM_SWEEP_ALLOW_PARTIAL=1,MDDM_SWEEP_TABLE_STEM=baseline_60_table", calls[1]["argv"])
        self.assertIn("--dependency=afterok:4101", calls[2]["argv"])
        self.assertIn("--dependency=afterok:4102", calls[3]["argv"])
        self.assertIn("--array=60-71%1", calls[3]["argv"])
        self.assertIn("--gpus=a100:1", calls[3]["argv"])
        self.assertIn("--dependency=afterok:4103", calls[4]["argv"])
        self.assertIn("--export=ALL,MDDM_SWEEP_ALLOW_PARTIAL=0,MDDM_SWEEP_TABLE_STEM=final_table", calls[4]["argv"])
        expected_checkpoint = str(output_root / "checkpoints/dpo_policy/model.safetensors")
        self.assertTrue(all(call["dpo"] == expected_checkpoint for call in calls))
        self.assertTrue(all(call["base"] == "base-revision-sha" for call in calls))
        self.assertTrue(all(call["dparallel"] == "dparallel-revision-sha" for call in calls))
        self.assertIn("Submitted fail-closed final 72-row table: 4104", completed.stdout)


if __name__ == "__main__":
    unittest.main()
