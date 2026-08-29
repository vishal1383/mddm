from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is installed by the Unity bootstrap, not system Python")
class DpoTrainingTest(unittest.TestCase):
    def test_collection_serialization_and_log_probability_gradient(self) -> None:
        from train_dpo_policy import (
            atomic_torch_save,
            collect_behavior_group,
            trajectory_behavior_cloning_loss,
            trajectory_log_probability,
        )

        class Output:
            def __init__(self, logits):
                self.logits = logits

        class FrozenModel:
            def __call__(self, input_ids, use_cache=False):
                logits = torch.zeros((*input_ids.shape, 7), device=input_ids.device)
                logits[..., 2] = 2.0
                return Output(logits)

        canvases, nfe, traces, behaviors = collect_behavior_group(
            FrozenModel(),
            [1],
            (0.5, 0.99),
            (2, 4),
            feature_width=3,
            mask_token_id=6,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(canvases), 4)
        self.assertLess(nfe[0], nfe[1])
        self.assertEqual(tuple(traces[0]["confidence"].shape[1:]), (256, 3))
        self.assertEqual(behaviors[-1], {"kind": "topk_wave", "value": 4})

        record = {"traces": traces, "pairs": [(0, 1)]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.pt"
            atomic_torch_save(path, record)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(loaded["pairs"], [(0, 1)])

        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, masked, confidence, timestep):
                return self.scale * confidence[..., 0]

        policy = TinyPolicy()
        logp = trajectory_log_probability(policy, traces[0], torch.device("cpu"), chunk=16)
        (-logp).backward()
        self.assertIsNotNone(policy.scale.grad)
        self.assertTrue(torch.isfinite(policy.scale.grad))

        policy.zero_grad(set_to_none=True)
        bc_loss = trajectory_behavior_cloning_loss(policy, traces[0], torch.device("cpu"), chunk=16)
        bc_loss.backward()
        self.assertTrue(torch.isfinite(policy.scale.grad))


if __name__ == "__main__":
    unittest.main()
