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
    @staticmethod
    def _wrapper(architecture):
        import sys

        sys.path.insert(0, str(Path(POLICY_REPO).resolve()))
        from common.models.policy import DiTConfidencePolicy, PolicyHFWrapper

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
        return PolicyHFWrapper(core, "dit_confidence")

    def test_official_bl32_checkpoint_round_trip(self) -> None:
        from evaluate import APPLE_POLICY_ARCHITECTURE, load_policy
        from safetensors.torch import save_file

        architecture = APPLE_POLICY_ARCHITECTURE
        wrapper = self._wrapper(architecture)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.safetensors"
            save_file(wrapper.state_dict(), str(checkpoint))
            loaded, receipt = load_policy(
                SimpleNamespace(
                    method="apple_policy_rl",
                    policy_repo=POLICY_REPO,
                    resolved_policy_architecture=architecture,
                    resolved_policy_checkpoint=checkpoint,
                    resolved_policy_checkpoint_receipt={"sha256": "fixture"},
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

    def test_hidden_state_dpo_checkpoint_round_trip(self) -> None:
        from evaluate import DPO_POLICY_ARCHITECTURE, ProjectedHiddenSetPolicy, load_policy
        from safetensors.torch import save_file

        architecture = {**DPO_POLICY_ARCHITECTURE, "base_hidden_dim": 16}
        wrapper = ProjectedHiddenSetPolicy(architecture)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.safetensors"
            save_file(wrapper.state_dict(), str(checkpoint))
            loaded, _receipt = load_policy(
                SimpleNamespace(
                    method="dpo_policy_v3",
                    policy_repo=POLICY_REPO,
                    resolved_policy_architecture=architecture,
                    resolved_policy_checkpoint=checkpoint,
                    resolved_policy_checkpoint_receipt={"sha256": "fixture"},
                ),
                torch.device("cpu"),
            )
        projected = loaded.project_hidden(torch.ones((2, 32, 16)))
        output = loaded(
            torch.ones((2, 32), dtype=torch.bool),
            projected,
            torch.zeros((2, 1)),
        )
        self.assertEqual(tuple(output.shape), (2, 32))


if __name__ == "__main__":
    unittest.main()
