from __future__ import annotations

import unittest

from dpo_objective import frontier_preference_pairs


class DpoObjectiveTest(unittest.TestCase):
    def test_fast_wrong_never_beats_slow_correct(self) -> None:
        winner, pairs = frontier_preference_pairs([False, True], [1, 32])
        self.assertEqual(winner, 1)
        self.assertEqual(pairs, [(1, 0, "safety")])

    def test_balances_hard_safety_and_efficiency_pairs(self) -> None:
        winner, pairs = frontier_preference_pairs([False, True, True, False], [5, 8, 20, 2])
        self.assertEqual(winner, 1)
        self.assertEqual(pairs, [(1, 3, "safety"), (1, 2, "efficiency")])

    def test_no_correct_path_produces_no_fabricated_pair(self) -> None:
        self.assertEqual(frontier_preference_pairs([False, False], [3, 9]), (None, []))


if __name__ == "__main__":
    unittest.main()
