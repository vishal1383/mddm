from __future__ import annotations

import unittest

from dpo_objective import multiplicative_reward, strict_preference_pairs


class DpoObjectiveTest(unittest.TestCase):
    def test_multiplicative_reward_never_prefers_fast_incorrect_paths(self) -> None:
        self.assertEqual(multiplicative_reward(False, 1, 256, 1.0), 0.0)
        self.assertGreater(multiplicative_reward(True, 16, 256, 1.0), 0.0)
        self.assertGreater(
            multiplicative_reward(True, 8, 256, 1.0),
            multiplicative_reward(True, 16, 256, 1.0),
        )

    def test_pairs_are_strict_and_reward_ordered(self) -> None:
        rewards = [0.0, 0.8, 0.8, 0.9]
        pairs = strict_preference_pairs(rewards)
        self.assertNotIn((1, 2), pairs)
        self.assertNotIn((2, 1), pairs)
        self.assertIn((3, 1), pairs)
        self.assertIn((1, 0), pairs)
        self.assertTrue(all(rewards[chosen] > rewards[rejected] for chosen, rejected in pairs))


if __name__ == "__main__":
    unittest.main()
