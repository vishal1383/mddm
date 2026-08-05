import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from Token2Token.core import (
    Commit,
    Target,
    anchor_loss,
    position_priors,
    relative_order_loss,
    select_targets,
)
from Token2Token.train import _token_ids, greedy_ig_targets
from Token2Token.decode import confidence_decode
from Token2Token.eval_gsm8k import (
    batch_confidence_decode,
    extract_gsm8k_answer,
    parse_k_values,
)
from Token2Token.eval_lm1b_loss import parse_mask_ratios
from Token2Token.precompute_rollout_targets import greedy_rollout_targets
from Token2Token.train_standard import masked_denoising_loss
from Token2Token.summarize_gsm8k_sweep import build_rows, render_report
from Token2Token.select_best_epoch import summarize_epochs
from Token2Token.train_anchor_order import (
    anchor_target_loss,
    anchor_completion_losses,
    completed_anchor_canvas,
    ordered_anchor_canvases,
    validate_target_provenance,
    validate_targets,
)


class ToyTokenizer:
    all_special_ids = []

    def decode(self, token_ids):
        return str(token_ids[0])


class ToyIGModel(torch.nn.Module):
    def forward(self, input_ids, use_cache=False):
        del use_cache
        rows = []
        for input_row in input_ids:
            canvas = input_row[1:]
            strength = (
                2.0 * float((canvas == 1).any())
                + 4.0 * float((canvas == 2).any())
                + 1.0 * float((canvas == 3).any())
            )
            logits = torch.zeros(len(input_row), 10)
            logits[:, 0] = strength
            rows.append(logits)
        return SimpleNamespace(logits=torch.stack(rows))


class ToyDecodeModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        logits = torch.zeros(len(input_ids), input_ids.shape[1], 10)
        for position in range(1, input_ids.shape[1]):
            logits[:, position, position] = float(position)
        return SimpleNamespace(logits=logits)


class ToyRolloutModel(torch.nn.Module):
    def forward(self, input_ids, use_cache=False):
        del use_cache
        gold = [1, 2, 3]
        rows = []
        for input_row in input_ids:
            canvas = input_row[1:]
            helpful_anchor = int(canvas[0]) == gold[0]
            logits = torch.zeros(len(input_row), 6)
            for position in range(len(canvas)):
                predicted = gold[position] if helpful_anchor else 5
                logits[position + 1, predicted] = 10.0
            rows.append(logits)
        return SimpleNamespace(logits=torch.stack(rows))


class CoreTests(unittest.TestCase):
    def test_target_selection_excludes_far_right_tail(self):
        targets = select_targets(
            list(range(20)),
            ToyTokenizer(),
            count=5,
            max_right_fraction=0.75,
            mask_token_id=99,
        )
        self.assertEqual(len(targets), 5)
        self.assertLessEqual(max(target.gold_position for target in targets), 14)

    def test_token_ids_accepts_current_transformers_shapes(self):
        self.assertEqual(_token_ids({"input_ids": [[1, 2, 3]]}), [1, 2, 3])

    def test_gsm8k_answer_extraction(self):
        self.assertEqual(extract_gsm8k_answer("work\n#### 2,640"), "2640")
        self.assertEqual(extract_gsm8k_answer("The answer is -3.5"), "-3.5")

    def test_k_value_parsing(self):
        self.assertEqual(parse_k_values("10,1,5,2,5"), [10, 1, 5, 2])

    def test_mask_ratio_parsing(self):
        self.assertEqual(parse_mask_ratios("0.25,1"), [0.25, 1.0])
        with self.assertRaises(ValueError):
            parse_mask_ratios("0")

    def test_greedy_ig_targets_are_selected_conditionally(self):
        candidates = [
            Target(0, 1, "one", 0),
            Target(1, 2, "two", 1),
            Target(2, 3, "three", 2),
        ]
        targets, scores = greedy_ig_targets(
            ToyIGModel(),
            [9],
            [1, 2, 3],
            candidates,
            0,
            count=3,
            batch_size=2,
            device="cpu",
        )
        self.assertEqual([target.gold_position for target in targets], [1, 0, 2])
        self.assertEqual([target.index for target in targets], [0, 1, 2])
        self.assertEqual(len(scores), 3)

    def test_rollout_targets_maximize_correct_standard_decodes(self):
        candidates = [
            Target(0, 1, "one", 0),
            Target(1, 2, "two", 1),
            Target(2, 3, "three", 2),
        ]
        targets, scores = greedy_rollout_targets(
            ToyRolloutModel(),
            [9],
            [1, 2, 3],
            candidates,
            0,
            count=1,
            rollout_steps=2,
            rollout_k=1,
            batch_size=3,
            device="cpu",
        )
        self.assertEqual(targets[0].gold_position, 0)
        self.assertEqual(scores[0], {"correct": 2, "committed": 2})

    def test_standard_confidence_decode(self):
        canvas = [0, 0, 0, 0]
        confidence_decode(
            ToyDecodeModel(),
            ToyTokenizer(),
            [9],
            canvas,
            0,
            steps=1,
            device="cpu",
        )
        self.assertNotIn(0, canvas)

    def test_fixed_top_k_confidence_decode(self):
        canvas = [0, 0, 0, 0]
        trace = confidence_decode(
            ToyDecodeModel(),
            ToyTokenizer(),
            [9],
            canvas,
            0,
            tokens_per_step=2,
            device="cpu",
        )
        self.assertEqual(len(trace), 2)
        self.assertEqual([len(item["filled"]) for item in trace], [2, 2])

    def test_batched_top_k_confidence_decode(self):
        canvases = batch_confidence_decode(
            ToyDecodeModel(),
            [[9], [8, 9]],
            4,
            0,
            tokens_per_step=2,
            device="cpu",
            pad_token_id=7,
        )
        self.assertEqual(len(canvases), 2)
        self.assertTrue(all(0 not in canvas for canvas in canvases))

    def test_masked_denoising_loss(self):
        positions = torch.tensor([0, 2])
        labels = torch.tensor([1, 3])
        correct = torch.zeros(3, 5)
        wrong = torch.zeros(3, 5)
        correct[0, 1] = 5.0
        correct[2, 3] = 5.0
        wrong[0, 4] = 5.0
        wrong[2, 4] = 5.0
        self.assertLess(
            float(masked_denoising_loss(correct, positions, labels)),
            float(masked_denoising_loss(wrong, positions, labels)),
        )

    def test_sweep_summary(self):
        summaries = {
            "anchor_lora": {1: {"accuracy": 0.5, "correct": 5, "examples": 10}},
            "standard_lora": {1: {"accuracy": 0.4, "correct": 4, "examples": 10}},
            "base": {1: {"accuracy": 0.3, "correct": 3, "examples": 10}},
        }
        rows = build_rows(summaries, [1, 2])
        self.assertAlmostEqual(rows[0]["anchor_minus_base_pp"], 20.0)
        self.assertIsNone(rows[1]["anchor_lora"]["accuracy"])
        report = render_report(rows)
        self.assertIn("50.00% (5/10)", report)
        self.assertIn("pending", report)

    def test_epoch_loss_summary(self):
        rows = [
            {"loss": 3.0, "anchor_loss": 1.0, "sequence_loss": 2.0},
            {"loss": 1.0, "anchor_loss": 0.5, "sequence_loss": 0.5},
            {"loss": 2.0, "anchor_loss": 1.5, "sequence_loss": 0.5},
            {"loss": 2.0, "anchor_loss": 1.0, "sequence_loss": 1.0},
        ]
        summaries = summarize_epochs(rows, 2, Path("run"), step_offset=4)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["epoch"], 3)
        self.assertEqual(summaries[0]["mean_loss"], 2.0)
        self.assertEqual(summaries[1]["adapter_path"], "run/checkpoint-000008")

    def test_plain_anchor_target_loss(self):
        correct = torch.zeros(3, 5)
        wrong = torch.zeros(3, 5)
        correct[1, 2] = 5.0
        wrong[1, 4] = 5.0
        self.assertLess(
            float(anchor_target_loss(correct, 1, 2)),
            float(anchor_target_loss(wrong, 1, 2)),
        )

    def test_anchor_completion_loss_supervises_remaining_masks(self):
        final_canvas = [1, 2, 0]
        positions = [0, 1]
        token_ids = [1, 2]
        gold_ids = [1, 2, 3]
        correct = torch.zeros(3, 3, 5)
        wrong = torch.zeros(3, 3, 5)
        correct[0, 0, 1] = 5.0
        correct[1, 1, 2] = 5.0
        correct[2, 2, 3] = 5.0
        wrong[0, 0, 4] = 5.0
        wrong[1, 1, 4] = 5.0
        wrong[2, 2, 4] = 5.0
        correct_anchor, correct_sequence = anchor_completion_losses(
            correct, final_canvas, positions, token_ids, gold_ids, 0
        )
        wrong_anchor, wrong_sequence = anchor_completion_losses(
            wrong, final_canvas, positions, token_ids, gold_ids, 0
        )
        self.assertLess(float(correct_anchor), float(wrong_anchor))
        self.assertLess(float(correct_sequence), float(wrong_sequence))

    def test_anchor_targets_must_follow_gold(self):
        validate_targets(
            [7, 8, 9],
            [
                {"rank": 1, "gold_position": 2, "token_id": 9},
                {"rank": 2, "gold_position": 0, "token_id": 7},
            ],
        )
        with self.assertRaises(ValueError):
            validate_targets(
                [7, 8, 9],
                [{"rank": 1, "gold_position": 2, "token_id": 8}],
            )

    def test_ordered_anchor_canvases(self):
        canvases, positions, token_ids = ordered_anchor_canvases(
            4,
            0,
            [
                {"gold_position": 2, "token_id": 7},
                {"gold_position": 0, "token_id": 8},
            ],
        )
        self.assertEqual(canvases, [[0, 0, 0, 0], [0, 0, 7, 0]])
        self.assertEqual(positions, [2, 0])
        self.assertEqual(token_ids, [7, 8])
        self.assertEqual(
            completed_anchor_canvas(canvases, positions, token_ids),
            [8, 0, 7, 0],
        )

    def test_anchor_trainer_requires_frozen_base_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.jsonl"
            metadata = path.with_suffix(".config.json")
            metadata.write_text(
                json.dumps({"target_source": "frozen_base_greedy_ig"}),
                encoding="utf-8",
            )
            validate_target_provenance(path, [{"target_source": None}])
            metadata.write_text(
                json.dumps(
                    {"target_source": "frozen_base_confidence_rollout"}
                ),
                encoding="utf-8",
            )
            validate_target_provenance(
                path, [{"target_source": "frozen_base_confidence_rollout"}]
            )
            metadata.write_text(
                json.dumps({"target_source": "v1_online_ig_recovered"}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_target_provenance(path, [{}])

    def test_grounding_prefers_gold_token(self):
        targets = [Target(0, 2, "correct", 1)]
        priors = position_priors(
            targets, {}, 3, sigma=1.0, max_right_fraction=1.0, device="cpu"
        )
        correct = torch.zeros(3, 6)
        wrong = torch.zeros(3, 6)
        correct[1, 2] = 5.0
        wrong[1, 4] = 5.0
        correct_loss, _, _ = anchor_loss(correct, targets, {}, priors)
        wrong_loss, _, _ = anchor_loss(wrong, targets, {}, priors)
        self.assertLess(
            float(correct_loss.grounding), float(wrong_loss.grounding)
        )

    def test_prior_peaks_between_neighbors_with_soft_order_tails(self):
        targets = [Target(0, 1, "a", 1), Target(1, 2, "b", 4), Target(2, 3, "c", 7)]
        commits = {0: Commit(0, 1, 1.0), 2: Commit(2, 7, 1.0)}
        priors = position_priors(
            targets, commits, 10, sigma=1.0, max_right_fraction=1.0, device="cpu"
        )
        self.assertEqual(int(torch.argmax(priors[1])), 4)
        self.assertGreater(float(priors[1, 0]), 0.0)
        self.assertGreater(float(priors[1, 8]), 0.0)
        self.assertEqual(float(priors[1, 1]), 0.0)
        self.assertEqual(float(priors[1, 7]), 0.0)
        self.assertAlmostEqual(float(priors[1].sum()), 1.0, places=6)

    def test_correct_order_has_lower_loss(self):
        correct = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        reversed_q = correct.flip(0)
        gold = torch.tensor([0, 1])
        self.assertLess(
            float(relative_order_loss(correct, gold)),
            float(relative_order_loss(reversed_q, gold)),
        )

    def test_anchor_loss_is_differentiable(self):
        targets = [Target(0, 2, "x", 1), Target(1, 3, "y", 4)]
        logits = torch.zeros(6, 8, requires_grad=True)
        logits.data[1, 2] = 5.0
        logits.data[4, 3] = 5.0
        priors = position_priors(
            targets, {}, 6, sigma=0.75, max_right_fraction=1.0, device="cpu"
        )
        losses, _, _ = anchor_loss(logits, targets, {}, priors)
        losses.total.backward()
        self.assertTrue(torch.isfinite(losses.total))
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
