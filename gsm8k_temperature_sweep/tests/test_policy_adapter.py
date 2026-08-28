from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


POLICY_REPO = os.environ.get("ML_RL_DLLM_REPO")


@unittest.skipIf(torch is None, "PyTorch is installed by the Unity bootstrap, not system Python")
@unittest.skipUnless(POLICY_REPO and Path(POLICY_REPO).is_dir(), "set ML_RL_DLLM_REPO to test the upstream adapter")
class PolicyAdapterTest(unittest.TestCase):
    def test_official_bl32_checkpoint_round_trip(self) -> None:
        import sys

        sys.path.insert(0, str(Path(POLICY_REPO).resolve()))
        from common.models.policy import DiTConfidencePolicy, PolicyHFWrapper
        from evaluate import PAPER_POLICY_ARCHITECTURE, load_policy
        from safetensors.torch import save_file

        architecture = PAPER_POLICY_ARCHITECTURE
        core = DiTConfidencePolicy(
            hidden_dim=architecture["hidden_dim"],
            feedforward_dim=architecture["feedforward_dim"],
            num_heads=architecture["num_heads"],
            dropout=architecture["dropout"],
            time_embed_dim=architecture["time_embed_dim"],
            confidences_top_p=architecture["confidences_top_p"],
            smart_init=architecture["smart_init"],
            num_blocks=architecture["num_blocks"],
            time_period=architecture["time_period"],
        )
        wrapper = PolicyHFWrapper(core, "dit_confidence")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.safetensors"
            save_file(wrapper.state_dict(), str(checkpoint))
            loaded, receipt = load_policy(
                SimpleNamespace(
                    method="paper_policy",
                    policy_repo=POLICY_REPO,
                    policy_checkpoint=str(checkpoint),
                ),
                torch.device("cpu"),
            )
        output = loaded(
            torch.ones((2, 32), dtype=torch.bool),
            torch.full((2, 32, 1), 0.8),
            torch.zeros((2, 1)),
        )
        self.assertEqual(tuple(output.shape), (2, 32))
        self.assertEqual(len(receipt["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
