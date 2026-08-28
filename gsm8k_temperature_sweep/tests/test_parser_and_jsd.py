from __future__ import annotations

import unittest

from answers import extract_gsm8k_answer

try:
    import torch
except ModuleNotFoundError:
    torch = None


class ParserAndJsdTest(unittest.TestCase):
    def test_answer_parser_prefers_box(self) -> None:
        self.assertEqual(extract_gsm8k_answer(r"intermediate 12; therefore \boxed{1,024}."), "1024")
        self.assertEqual(extract_gsm8k_answer("work\n#### 37"), "37")

    @unittest.skipIf(torch is None, "PyTorch is installed by the Unity bootstrap, not system Python")
    def test_jsd_selector_is_deterministic_and_candidate_bounded(self) -> None:
        from evaluate import distribution_interaction_positions

        torch.manual_seed(4)
        logits = torch.randn(1, 4, 7)
        candidates = torch.tensor([[False, True, True, False]])
        left = distribution_interaction_positions(logits, candidates, mask_token_id=6)
        right = distribution_interaction_positions(logits, candidates, mask_token_id=6)
        self.assertEqual(left, right)
        self.assertTrue(set(left[0]).issubset({1, 2}))


if __name__ == "__main__":
    unittest.main()
