from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
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
        self.assertIn('cd "$SLURM_SUBMIT_DIR"', batch)
        self.assertIn('EXPERIMENT_ROOT="$PWD"', batch)
        self.assertNotIn('dirname "${BASH_SOURCE[0]}"', batch)
        self.assertNotIn("SFT_ADAPTER_PATH=", batch)
        self.assertNotIn("UNMASKING_POLICY_CHECKPOINT=", batch)
        self.assertNotIn("Export SFT_ADAPTER_PATH before sbatch", batch)
        self.assertNotIn("Export UNMASKING_POLICY_CHECKPOINT before sbatch", batch)
        self.assertNotIn("Export HF_TOKEN before sbatch", batch)

    def test_spooled_batch_resolves_the_real_submission_directory(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            submit_dir = temporary / "checkout/gsm8k_temperature_sweep"
            scripts = submit_dir / "scripts"
            scripts.mkdir(parents=True)
            (scripts / "bootstrap_env.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (scripts / "run_sequential.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            spool = temporary / "var/spool/slurm/slurmd/job123/slurm_script"
            spool.parent.mkdir(parents=True)
            shutil.copyfile(root / "slurm/submit_all.sbatch", spool)
            fake_bin = temporary / "bin"
            fake_bin.mkdir()
            calls = temporary / "bash-calls.txt"
            fake_bash = fake_bin / "bash"
            fake_bash.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$1" >> "$FAKE_BASH_CALLS"\n', encoding="utf-8"
            )
            fake_bash.chmod(0o755)
            completed = subprocess.run(
                ["/bin/bash", str(spool)],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "SLURM_SUBMIT_DIR": str(submit_dir),
                    "FAKE_BASH_CALLS": str(calls),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                [str(scripts / "bootstrap_env.sh"), str(scripts / "run_sequential.sh")],
            )


if __name__ == "__main__":
    unittest.main()
