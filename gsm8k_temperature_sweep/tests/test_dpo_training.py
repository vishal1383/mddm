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
            collect_hidden_policy_group,
            trajectory_log_probability,
        )

        class Output:
            def __init__(self, logits, hidden):
                self.logits = logits
                self.hidden_states = (hidden,)

        class FrozenModel:
            def __call__(self, input_ids, use_cache=False, output_hidden_states=False):
                logits = torch.zeros((*input_ids.shape, 7), device=input_ids.device)
                logits[..., 2] = 2.0
                hidden = torch.ones((*input_ids.shape, 3), device=input_ids.device)
                return Output(logits, hidden)

        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.1))

            def project_hidden(self, hidden):
                return hidden

            def forward(self, masked, hidden, timestep):
                return self.scale * hidden[..., 0]

        policy = TinyPolicy()
        canvases, nfe, traces = collect_hidden_policy_group(
            FrozenModel(),
            policy,
            [1],
            (-2.0, 2.0),
            proposal_temperature=1.0,
            example_index=0,
            master_seed=42,
            mask_token_id=6,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(canvases), 2)
        self.assertLess(nfe[1], nfe[0])
        self.assertEqual(tuple(traces[0]["hidden"].shape[1:]), (256, 3))

        class DifferentTokenModel(FrozenModel):
            def __call__(self, input_ids, use_cache=False, output_hidden_states=False):
                output = super().__call__(input_ids, use_cache, output_hidden_states)
                output.logits.zero_()
                output.logits[..., 3] = 2.0
                return output

        _other_canvases, other_nfe, other_traces = collect_hidden_policy_group(
            DifferentTokenModel(),
            policy,
            [1],
            (-2.0, 2.0),
            proposal_temperature=1.0,
            example_index=0,
            master_seed=42,
            mask_token_id=6,
            device=torch.device("cpu"),
        )
        self.assertEqual(nfe, other_nfe)
        for trace, other in zip(traces, other_traces):
            self.assertTrue(torch.equal(trace["action"], other["action"]))

        record = {"traces": traces, "pairs": [(0, 1)]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.pt"
            atomic_torch_save(path, record)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(loaded["pairs"], [(0, 1)])

        logp = trajectory_log_probability(policy, traces[0], torch.device("cpu"), chunk=16)
        (-logp).backward()
        self.assertIsNotNone(policy.scale.grad)
        self.assertTrue(torch.isfinite(policy.scale.grad))


if __name__ == "__main__":
    unittest.main()
