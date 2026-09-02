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
            'seq 0 19',
            'Stage 2/5:',
            'baseline_table',
            'Stage 3/5:',
            'train_dpo_policy.py',
            'Stage 4/5:',
            'seq 20 23',
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
        self.assertNotIn("SLURM_SUBMIT_DIR", batch)
        self.assertNotIn("EXPERIMENT_ROOT=", batch)
        self.assertIn("cd gsm8k_temperature_sweep", batch)
        self.assertIn('MDDM_SWEEP_STATE_ROOT="${MDDM_SWEEP_STATE_ROOT:-$PWD/.runtime}"', batch)
        self.assertNotIn("SCRATCH", batch)
        self.assertIn('PIP_CACHE_DIR="$MDDM_SWEEP_STATE_ROOT/pip-cache"', batch)
        self.assertIn('TMPDIR="$MDDM_SWEEP_STATE_ROOT/tmp"', batch)
        self.assertIn('HF_HOME="$MDDM_SWEEP_STATE_ROOT/huggingface"', batch)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", batch)
        bootstrap = (root / "scripts/bootstrap_env.sh").read_text(encoding="utf-8")
        self.assertIn("status --porcelain --untracked-files=no", bootstrap)
        self.assertIn(".mddm-requirements-sha256", bootstrap)
        self.assertIn("cat-file -e", bootstrap)
        self.assertIn("ml-rl-dllm-35e4830485f1", bootstrap)
        self.assertTrue((root / "final_results/manifests/.gitkeep").is_file())
        self.assertIn(
            "#SBATCH --output=gsm8k_temperature_sweep/final_results/manifests/slurm-%j.out",
            batch,
        )
        self.assertIn("bash scripts/bootstrap_env.sh", batch)
        self.assertIn("exec bash scripts/run_sequential.sh", batch)
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
            checkout = temporary / "checkout"
            submit_dir = checkout / "gsm8k_temperature_sweep"
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
                    "FAKE_BASH_CALLS": str(calls),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                },
                cwd=checkout,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["scripts/bootstrap_env.sh", "scripts/run_sequential.sh"],
            )


if __name__ == "__main__":
    unittest.main()
