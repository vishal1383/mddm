from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiment_contract import file_sha256
from train_apple_policy import completed_checkpoint, latest_resumable_checkpoint


class AppleTrainingResumeTest(unittest.TestCase):
    def test_ignores_incomplete_periodic_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root / "checkpoint-25"
            complete.mkdir()
            for name in ("model.safetensors", "trainer_state.json", "optimizer.pt", "scheduler.pt"):
                (complete / name).write_bytes(b"ok")
            incomplete = root / "checkpoint-50"
            incomplete.mkdir()
            (incomplete / "model.safetensors").write_bytes(b"partial")
            self.assertEqual(latest_resumable_checkpoint(root), complete)

    def test_accepts_only_hash_sealed_completed_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "checkpoint-best" / "model.safetensors"
            selected.parent.mkdir()
            selected.write_bytes(b"weights")
            manifest = root / "training_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "selected_checkpoint_sha256": file_sha256(selected),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(completed_checkpoint(root), selected)
            selected.write_bytes(b"changed")
            self.assertIsNone(completed_checkpoint(root))


if __name__ == "__main__":
    unittest.main()
