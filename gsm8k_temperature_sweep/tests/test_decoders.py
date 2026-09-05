from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is installed by the Unity bootstrap, not system Python")
class DecoderTransitionTest(unittest.TestCase):
    def test_every_method_finishes_with_one_model_call_per_cycle(self) -> None:
        from evaluate import decode_batch

        class Output:
            def __init__(self, logits):
                self.logits = logits
                self.hidden_states = (torch.ones((*logits.shape[:2], 4), device=logits.device),)

        class Model:
            def __call__(self, input_ids, use_cache=False, output_hidden_states=False):
                logits = torch.zeros((*input_ids.shape, 11), device=input_ids.device)
                logits[..., 2] = 12.0
                return Output(logits)

        class Policy:
            def project_hidden(self, hidden):
                return hidden

            def __call__(self, masked, features, timestep):
                return torch.full(masked.shape, 20.0, device=masked.device)

        for method in ("base", "jsd_mean_field", "dparallel", "justgrpo", "lora_sft", "apple_policy_rl", "dpo_policy_v3"):
            with self.subTest(method=method):
                result = decode_batch(
                    Model(),
                    [1],
                    method=method,
                    policy=Policy() if method in {"apple_policy_rl", "dpo_policy_v3"} else None,
                    temperature=0.5,
                    policy_temperature=0.5,
                    confidence_threshold=0.9,
                    entropy_threshold=0.5,
                    canvas_length=4,
                    block_length=2,
                    samples=2,
                    example_index=0,
                    seed=42,
                    mask_token_id=6,
                    device=torch.device("cpu"),
                )
                self.assertEqual(result["canvases"], [[2, 2, 2, 2], [2, 2, 2, 2]])
                self.assertEqual(result["nfe"], [2, 2])
                self.assertEqual(len(result["trace_sha256"]), 2)


if __name__ == "__main__":
    unittest.main()
