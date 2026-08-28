from __future__ import annotations

from pathlib import Path
import unittest


class SubmissionChainTest(unittest.TestCase):
    def test_baselines_then_dpo_then_final_table_dependency_order(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts/submit.sh").read_text(encoding="utf-8")
        markers = (
            'BASELINE_ARRAY_JOB_ID="$(sbatch',
            '--array="0-59%$ARRAY_LIMIT"',
            'BASELINE_TABLE_JOB_ID="$(sbatch',
            '--dependency="afterok:$BASELINE_ARRAY_JOB_ID"',
            'DPO_TRAIN_JOB_ID="$(sbatch',
            '--dependency="afterok:$BASELINE_TABLE_JOB_ID"',
            'DPO_ARRAY_JOB_ID="$(sbatch',
            '--dependency="afterok:$DPO_TRAIN_JOB_ID"',
            '--array="60-71%$ARRAY_LIMIT"',
            'FINAL_TABLE_JOB_ID="$(sbatch',
            '--dependency="afterok:$DPO_ARRAY_JOB_ID"',
        )
        offsets = [script.index(marker) for marker in markers]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn('MDDM_SWEEP_GPUS:-a100:1', script)
        self.assertIn('ARRAY_LIMIT="${MDDM_SWEEP_ARRAY_LIMIT:-1}"', script)


if __name__ == "__main__":
    unittest.main()
