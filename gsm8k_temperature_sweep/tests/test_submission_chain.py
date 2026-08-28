from __future__ import annotations

from pathlib import Path
import unittest


class SubmissionChainTest(unittest.TestCase):
    def test_one_a100_batch_runs_every_stage_in_order_without_nested_sbatch(self) -> None:
        root = Path(__file__).resolve().parents[1]
        batch = (root / "slurm/submit_all.sbatch").read_text(encoding="utf-8")
        script = (root / "scripts/run_sequential.sh").read_text(encoding="utf-8")
        markers = (
            'Stage 1/5:',
            'seq 0 59',
            'Stage 2/5:',
            'baseline_60_table',
            'Stage 3/5:',
            'train_dpo_policy.py',
            'Stage 4/5:',
            'seq 60 71',
            'Stage 5/5:',
            'final_table',
        )
        offsets = [script.index(marker) for marker in markers]
        self.assertEqual(offsets, sorted(offsets))
        self.assertNotIn("sbatch", script)
        self.assertIn("#SBATCH --partition=gpu-preempt", batch)
        self.assertIn("#SBATCH --gres=gpu:a100:1", batch)
        self.assertNotIn("#SBATCH --partition=cpu", batch)
        self.assertNotIn("#SBATCH --qos=", batch)


if __name__ == "__main__":
    unittest.main()
