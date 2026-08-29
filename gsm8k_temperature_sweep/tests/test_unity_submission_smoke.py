from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class UnitySubmissionSmokeTest(unittest.TestCase):
    def test_in_allocation_launcher_executes_all_18_cells_in_order(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_venv = temporary / "venv/bin"
            fake_venv.mkdir(parents=True)
            calls_path = temporary / "python_calls.txt"
            fake_python = fake_venv / "python"
            fake_python.write_text(
                """#!/usr/bin/env bash
echo "$*" >> "$FAKE_PYTHON_CALLS"
if [[ "${1:-}" == *seal_model_revisions.py ]]; then
  echo base-revision-sha dparallel-revision-sha paper-policy-revision-sha
elif [[ "${1:-}" == *is_cell_complete.py ]]; then
  exit 1
fi
exit 0
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            output_root = temporary / "results"
            env = {
                **os.environ,
                "MDDM_SWEEP_VENV": str(temporary / "venv"),
                "ML_RL_DLLM_REPO": str(temporary / "policy-repo"),
                "MDDM_SWEEP_OUTPUT_ROOT": str(output_root),
                "FAKE_PYTHON_CALLS": str(calls_path),
            }
            completed = subprocess.run(
                ["bash", str(root / "scripts/run_sequential.sh")],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            calls = calls_path.read_text(encoding="utf-8").splitlines()

        eval_calls = [call for call in calls if "/evaluate.py " in call]
        task_ids = [int(call.split("--task-id ", 1)[1].split()[0]) for call in eval_calls]
        self.assertEqual(task_ids, list(range(18)))
        self.assertEqual(sum("/train_dpo_policy.py " in call for call in calls), 1)
        self.assertEqual(sum("/aggregate.py " in call for call in calls), 2)
        self.assertIn("Stage 1/5", completed.stdout)
        self.assertIn("Stage 5/5", completed.stdout)
        self.assertIn(str(output_root / "tables/final_table.md"), completed.stdout)


if __name__ == "__main__":
    unittest.main()
