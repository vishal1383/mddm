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
from Token2Token.decode import confidence_decode, threshold_unlock_decode
from Token2Token.eval_gsm8k import (
    batch_confidence_decode,
    extract_gsm8k_answer,
    parse_k_values,
)
from Token2Token.eval_lm1b_loss import parse_mask_ratios
from Token2Token.eval_threshold_gsm8k import (
    batch_threshold_unlock_decode,
    batch_topk_decode,
    limit_threshold_selection,
    parse_thresholds,
    threshold_tag,
)
from Token2Token.precompute_local_unlock_targets import (
    greedy_local_unlock_targets,
    shifted_window,
)
from Token2Token.precompute_rollout_targets import greedy_rollout_targets
from Token2Token.precompute_threshold_unlock_targets import (
    candidate_key,
    correct_threshold_positions,
    is_allowed_anchor_token,
    plausible_candidates,
    threshold_unlock_trajectory,
)
from Token2Token.train_standard import masked_denoising_loss
from Token2Token.train_anchor_transition import (
    anchor_transitions,
    masked_kl_loss,
    post_anchor_topk_positions,
)
from Token2Token.paired_comparison import compare, mcnemar_p_value
from Token2Token.train_parallel_unlock import (
    bucket_positions,
    gold_cross_entropy,
    preserve_kl,
    promote_objective,
)
from Token2Token.train_threshold_unlock import trajectory_stages
from Token2Token.summarize_gsm8k_sweep import build_rows, render_report
from Token2Token.select_best_epoch import summarize_epochs
from Token2Token.summarize_threshold_comparison import (
    add_latency_metrics,
    comparison,
    render_markdown as render_threshold_comparison,
)
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


class ToyTextTokenizer(ToyTokenizer):
    def decode(self, token_ids):
        return {1: "one", 2: " two", 3: "three"}.get(token_ids[0], "word")


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


class ToyThresholdModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        logits = torch.zeros(len(input_ids), input_ids.shape[1], 6)
        strengths = [8.0, 6.0, 1.0]
        for completion_position, strength in enumerate(strengths):
            sequence_position = completion_position + 1
            logits[:, sequence_position, completion_position + 1] = strength
        return SimpleNamespace(logits=logits)


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

    def test_local_unlock_targets_maximize_new_top1_gold_tokens(self):
        candidates = [
            Target(0, 1, "one", 0),
            Target(1, 2, "two", 1),
            Target(2, 3, "three", 2),
        ]
        targets, scores = greedy_local_unlock_targets(
            ToyRolloutModel(),
            [9],
            [1, 2, 3],
            candidates,
            0,
            count=1,
            window_size=3,
            batch_size=3,
            device="cpu",
        )
        self.assertEqual(targets[0].gold_position, 0)
        self.assertEqual(scores[0]["gain"], 2)

    def test_shifted_window_preserves_width_at_edges(self):
        self.assertEqual(shifted_window(0, 20, 9), (0, 9))
        self.assertEqual(shifted_window(19, 20, 9), (11, 20))

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

    def test_threshold_unlock_decode_uses_top1_then_threshold_burst(self):
        canvas = [0, 0, 0]
        trace = threshold_unlock_decode(
            ToyThresholdModel(),
            ToyTextTokenizer(),
            [9],
            canvas,
            0,
            confidence_threshold=0.95,
            device="cpu",
        )
        self.assertEqual(canvas, [1, 2, 3])
        self.assertEqual(
            [(item["phase"], len(item["filled"])) for item in trace],
            [("catalyst", 1), ("threshold_unlock", 1), ("catalyst", 1)],
        )

    def test_batched_threshold_unlock_decode(self):
        canvases, stats = batch_threshold_unlock_decode(
            ToyThresholdModel(),
            [[9], [8]],
            3,
            0,
            confidence_threshold=0.95,
            tokenizer=ToyTextTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(canvases, [[1, 2, 3], [1, 2, 3]])
        self.assertTrue(all(item["threshold_tokens"] == 1 for item in stats))
        self.assertEqual(parse_thresholds(".95,.9,.95"), [0.95, 0.9])
        self.assertEqual(threshold_tag(0.95), "t0p95")

    def test_threshold_selection_is_capped_by_confidence(self):
        selected = torch.tensor([[True, True, False, True]])
        confidence = torch.tensor([[0.96, 0.99, 1.0, 0.97]])
        limited = limit_threshold_selection(selected, confidence, 2)
        self.assertEqual(limited.tolist(), [[False, True, False, True]])

    def test_threshold_decode_uses_cleanup_for_non_text_predictions(self):
        canvases, stats = batch_threshold_unlock_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            confidence_threshold=0.95,
            tokenizer=ToyTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(canvases, [[1, 2, 3]])
        self.assertEqual(stats[0]["catalyst_tokens"], 0)
        self.assertEqual(stats[0]["cleanup_tokens"], 3)
        self.assertEqual(stats[0]["threshold_tokens"], 0)

    def test_first_forward_commit_fills_confident_cleanup_positions(self):
        canvases, stats = batch_threshold_unlock_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            confidence_threshold=0.95,
            commit_threshold_on_first_forward=True,
            tokenizer=ToyTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(canvases, [[1, 2, 3]])
        self.assertEqual(stats[0]["cleanup_tokens"], 2)
        self.assertEqual(stats[0]["first_forward_threshold_tokens"], 1)
        self.assertEqual(stats[0]["model_forwards"], 2)

    def test_single_forward_mode_skips_the_unlock_forward(self):
        canvases, stats = batch_threshold_unlock_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            confidence_threshold=0.95,
            commit_threshold_on_first_forward=True,
            unlock_forward=False,
            tokenizer=ToyTextTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(canvases, [[1, 2, 3]])
        self.assertEqual(stats[0]["cycles"], stats[0]["model_forwards"])

    def test_base_first_forward_disables_the_adapter_only_for_the_catalyst(self):
        from contextlib import contextmanager

        class GatedModel(ToyThresholdModel):
            def __init__(self):
                super().__init__()
                self.disabled_forwards = 0
                self.enabled_forwards = 0
                self.adapter_off = False

            @contextmanager
            def disable_adapter(self):
                self.adapter_off = True
                try:
                    yield
                finally:
                    self.adapter_off = False

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                if self.adapter_off:
                    self.disabled_forwards += 1
                else:
                    self.enabled_forwards += 1
                return super().forward(input_ids, attention_mask, use_cache)

        model = GatedModel()
        batch_threshold_unlock_decode(
            model,
            [[9]],
            3,
            0,
            confidence_threshold=0.95,
            base_first_forward=True,
            tokenizer=ToyTextTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        self.assertGreater(model.disabled_forwards, 0)
        self.assertGreater(model.enabled_forwards, 0)
        self.assertFalse(model.adapter_off)

    def test_any_catalyst_filter_removes_the_text_restriction(self):
        canvases, stats = batch_threshold_unlock_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            confidence_threshold=0.999,
            catalyst_filter="any",
            tokenizer=ToyTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(canvases, [[1, 2, 3]])
        # The text filter would send every one of these digit tokens to
        # cleanup; with the filter off they are ordinary catalysts.
        self.assertEqual(stats[0]["cleanup_tokens"], 0)
        self.assertEqual(stats[0]["catalyst_tokens"], 3)

    def test_multiple_catalysts_commit_in_one_forward(self):
        canvases, stats = batch_threshold_unlock_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            confidence_threshold=0.999,
            catalyst_tokens_per_forward=2,
            tokenizer=ToyTextTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(canvases, [[1, 2, 3]])
        self.assertEqual(stats[0]["catalyst_tokens"], 3)

    def test_single_forward_mode_requires_first_forward_commits(self):
        with self.assertRaises(ValueError):
            batch_threshold_unlock_decode(
                ToyThresholdModel(),
                [[9]],
                3,
                0,
                confidence_threshold=0.95,
                unlock_forward=False,
                tokenizer=ToyTokenizer(),
                device="cpu",
                pad_token_id=5,
            )

    def test_batched_topk_decode_places_k_tokens_per_forward(self):
        canvases, stats = batch_topk_decode(
            ToyThresholdModel(),
            [[9], [8]],
            3,
            0,
            tokens_per_step=2,
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(canvases, [[1, 2, 3], [1, 2, 3]])
        self.assertTrue(all(item["model_forwards"] == 2 for item in stats))

    def test_threshold_decode_cleanup_is_left_to_right(self):
        canvas = [0, 0, 0]
        trace = threshold_unlock_decode(
            ToyDecodeModel(),
            ToyTokenizer(),
            [9],
            canvas,
            0,
            confidence_threshold=0.95,
            device="cpu",
        )
        cleanup_positions = [
            item["filled"][0]["position"]
            for item in trace
            if item["phase"] == "cleanup"
        ]
        self.assertEqual(cleanup_positions, [0, 1, 2])

    def test_threshold_set_keeps_only_correct_high_confidence_predictions(self):
        unlocked = correct_threshold_positions(
            [0, 0, 7, 0, 0],
            [1, 2, 7, 4, 5],
            [1, 9, 7, 4, 8],
            [0.96, 0.99, 0.99, 0.951, 0.949],
            candidate_position=0,
            mask_token_id=0,
            threshold=0.95,
        )
        self.assertEqual([item["gold_position"] for item in unlocked], [3])

    def test_threshold_target_trajectory_places_catalyst_then_burst(self):
        rounds, residual = threshold_unlock_trajectory(
            ToyThresholdModel(),
            ToyTextTokenizer(),
            [9],
            [1, 2, 3],
            0,
            confidence_threshold=0.95,
            candidate_prob_ratio=0.5,
            candidate_batch_size=2,
            device="cpu",
        )
        self.assertEqual(len(rounds), 2)
        self.assertEqual(residual, [])
        self.assertEqual(rounds[0]["catalyst"]["gold_position"], 0)
        self.assertEqual([item["gold_position"] for item in rounds[0]["unlocked"]], [1])
        self.assertEqual(rounds[1]["catalyst"]["gold_position"], 2)

    def test_anchor_filter_rejects_whitespace_symbols_and_numbers(self):
        class MixedTokenizer:
            all_special_ids = [5]

            def decode(self, token_ids):
                return {
                    1: " ",
                    2: "<<",
                    3: "42",
                    4: " total",
                    5: "special",
                    6: "2nd",
                }[token_ids[0]]

        tokenizer = MixedTokenizer()
        self.assertEqual(
            [is_allowed_anchor_token(token_id, tokenizer) for token_id in range(1, 7)],
            [False, False, False, True, False, False],
        )
        rounds, residual = threshold_unlock_trajectory(
            ToyThresholdModel(),
            tokenizer,
            [9],
            [1, 2, 3],
            0,
            confidence_threshold=0.95,
            candidate_prob_ratio=0.5,
            candidate_batch_size=2,
            device="cpu",
        )
        self.assertEqual(rounds, [])
        self.assertEqual([item["gold_position"] for item in residual], [0, 1, 2])

    def test_threshold_candidate_uses_gain_then_after_count(self):
        log_probabilities = {0: -0.1, 1: -0.3, 2: -2.0}
        self.assertEqual(
            plausible_candidates([0, 1, 2], log_probabilities, 0.5), [0, 1]
        )
        larger_gain = {
            "position": 0,
            "gold_log_probability": -0.3,
            "correct_before": 0,
            "correct_after": [{}, {}],
        }
        larger_after_but_smaller_gain = {
            "position": 1,
            "gold_log_probability": -0.1,
            "correct_before": 2,
            "correct_after": [{}, {}, {}],
        }
        self.assertGreater(
            candidate_key(larger_gain), candidate_key(larger_after_but_smaller_gain)
        )
        after_three = {
            "position": 2,
            "gold_log_probability": -0.5,
            "correct_before": 1,
            "correct_after": [{}, {}, {}],
        }
        after_two = {
            "position": 3,
            "gold_log_probability": -0.1,
            "correct_before": 0,
            "correct_after": [{}, {}],
        }
        self.assertGreater(candidate_key(after_three), candidate_key(after_two))

    def test_threshold_trajectory_stages_fill_each_token_once(self):
        record = {
            "gold_ids": [1, 2, 3, 4],
            "rounds": [
                {
                    "round": 1,
                    "catalyst": {"gold_position": 1, "token_id": 2},
                    "unlocked": [
                        {"gold_position": 3, "token_id": 4},
                        {"gold_position": 0, "token_id": 1},
                    ],
                },
                {
                    "round": 2,
                    "catalyst": {"gold_position": 2, "token_id": 3},
                    "unlocked": [],
                },
            ],
            "residual": [],
        }
        stages = trajectory_stages(record, 0)
        self.assertEqual([item["kind"] for item in stages], ["catalyst", "catalyst"])
        self.assertEqual(stages[0]["canvas"], [0, 0, 0, 0])
        self.assertEqual(stages[1]["canvas"], [1, 2, 0, 4])

    def test_threshold_trajectory_stages_use_ltr_for_residual_only_record(self):
        record = {
            "gold_ids": [1, 2, 3],
            "rounds": [],
            "residual": [
                {"gold_position": 0, "token_id": 1},
                {"gold_position": 1, "token_id": 2},
                {"gold_position": 2, "token_id": 3},
            ],
        }
        stages = trajectory_stages(record, 0)
        self.assertEqual(
            [item["kind"] for item in stages],
            ["left_to_right_cleanup"] * 3,
        )
        self.assertEqual(
            [item["canvas"] for item in stages],
            [[0, 0, 0], [1, 0, 0], [1, 2, 0]],
        )
        self.assertEqual(
            [item["positions"] for item in stages],
            [[0], [1], [2]],
        )

    def test_anchor_transition_builds_post_anchor_canvas(self):
        record = {
            "gold_ids": [1, 2, 3, 4, 5],
            "rounds": [
                {
                    "round": 1,
                    "catalyst": {"gold_position": 2, "token_id": 3},
                    "unlocked": [
                        {"gold_position": 0, "token_id": 1, "confidence": 0.96},
                        {"gold_position": 4, "token_id": 5, "confidence": 0.99},
                        {"gold_position": 1, "token_id": 2, "confidence": 0.98},
                    ],
                },
                {
                    "round": 2,
                    "catalyst": {"gold_position": 3, "token_id": 4},
                    "unlocked": [],
                },
            ],
        }
        transitions = anchor_transitions(record, 0, 2)
        self.assertEqual(transitions[0]["anchor"]["canvas"], [0, 0, 0, 0, 0])
        self.assertEqual(
            transitions[0]["post_anchor"]["canvas"], [0, 0, 3, 0, 0]
        )
        self.assertIsNone(transitions[1]["post_anchor"])

    def test_post_anchor_topk_uses_current_model_confidence(self):
        logits = torch.zeros(4, 8)
        logits[0, 1] = 2.0
        logits[2, 2] = 5.0
        logits[3, 3] = 4.0
        self.assertEqual(
            post_anchor_topk_positions(logits, [0, 7, 0, 0], 0, 2),
            [2, 3],
        )

    def test_parallel_unlock_buckets_split_by_commit_decision(self):
        teacher = torch.zeros(4, 6)
        teacher[0, 1] = 8.0
        teacher[1, 2] = 1.8
        teacher[2, 5] = 8.0
        teacher[3, 5] = 1.8
        buckets = bucket_positions(
            teacher,
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            0,
            0.95,
            0.5,
        )
        self.assertEqual(buckets["promote"], [1])
        self.assertEqual(buckets["repair"], [2])
        self.assertEqual(buckets["preserve"], [0, 3])

    def test_numeric_positions_are_pinned_to_the_base_distribution(self):
        class DigitTokenizer:
            def decode(self, token_ids):
                return {1: " 67", 2: " dollars"}[token_ids[0]]

        teacher = torch.zeros(2, 6)
        teacher[0, 1] = 1.8
        teacher[1, 2] = 1.8
        unprotected = bucket_positions(teacher, [0, 0], [1, 2], 0, 0.95, 0.5)
        self.assertEqual(unprotected["promote"], [0, 1])
        protected = bucket_positions(
            teacher, [0, 0], [1, 2], 0, 0.95, 0.5, 0, DigitTokenizer(), {}
        )
        self.assertEqual(protected["promote"], [1])
        self.assertEqual(protected["preserve"], [0])

    def test_repair_skips_positions_where_gold_is_not_a_live_alternative(self):
        teacher = torch.zeros(1, 6)
        teacher[0, 5] = 8.0
        teacher[0, 3] = -8.0
        confident_wrong = bucket_positions(teacher, [0], [3], 0, 0.95, 0.5, 0)
        self.assertEqual(confident_wrong["repair"], [0])
        filtered = bucket_positions(teacher, [0], [3], 0, 0.95, 0.5, 2)
        self.assertEqual(filtered["repair"], [])
        self.assertEqual(filtered["preserve"], [0])

    def test_parallel_unlock_buckets_ignore_filled_positions(self):
        teacher = torch.zeros(2, 6)
        teacher[0, 1] = 1.8
        teacher[1, 2] = 1.8
        buckets = bucket_positions(teacher, [1, 0], [1, 2], 0, 0.95, 0.5)
        self.assertEqual(buckets["promote"], [1])
        self.assertEqual(buckets["preserve"], [])

    def test_parallel_unlock_losses_are_zero_without_positions(self):
        logits = torch.zeros(1, 2, 6, requires_grad=True)
        zero = logits.sum() * 0.0
        self.assertEqual(
            float(gold_cross_entropy(logits, [[]], [1, 2], zero).detach()), 0.0
        )
        self.assertEqual(
            float(preserve_kl(logits, logits, [[]], zero).detach()), 0.0
        )
        loss = gold_cross_entropy(logits, [[0]], [1, 2], zero)
        loss.backward()
        self.assertTrue(bool(logits.grad.abs().sum() > 0))

    def test_promote_hinge_stops_once_the_position_is_committable(self):
        committable = torch.zeros(1, 2, 6)
        committable[0, 0, 1] = 12.0
        zero = committable.sum() * 0.0
        self.assertEqual(
            float(promote_objective(committable, [[0]], [1, 2], zero, "hinge", 0.97)),
            0.0,
        )
        self.assertGreater(
            float(promote_objective(committable, [[0]], [1, 2], zero, "ce", 0.97)),
            0.0,
        )
        undecided = torch.zeros(1, 2, 6)
        undecided[0, 0, 1] = 1.8
        self.assertGreater(
            float(promote_objective(undecided, [[0]], [1, 2], zero, "hinge", 0.97)),
            0.0,
        )

    def test_masked_kl_preserves_base_distribution(self):
        teacher = torch.zeros(3, 5)
        identical = teacher.clone()
        changed = teacher.clone()
        changed[0, 1] = 4.0
        self.assertAlmostEqual(
            float(masked_kl_loss(identical, teacher, [0, 7, 0], 0)),
            0.0,
            places=6,
        )
        self.assertGreater(
            float(masked_kl_loss(changed, teacher, [0, 7, 0], 0)),
            0.0,
        )

    def test_mcnemar_uses_only_discordant_pairs(self):
        self.assertEqual(mcnemar_p_value(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_p_value(4, 2), 0.6875)
        self.assertAlmostEqual(mcnemar_p_value(2, 4), 0.6875)
        self.assertLess(mcnemar_p_value(10, 0), 0.01)
        # Concordant pairs never enter the test, so a large shared correct
        # count cannot make a small discordant split look significant.
        self.assertEqual(mcnemar_p_value(1, 1), 1.0)

    def test_paired_comparison_counts_outcomes_and_forward_deltas(self):
        def rows(flags, forwards):
            return [
                {"example_id": str(i), "correct": flag, "model_forwards": count}
                for i, (flag, count) in enumerate(zip(flags, forwards))
            ]

        result = compare(
            rows([True, True, False, False], [100, 100, 100, 100]),
            rows([True, False, True, False], [90, 90, 90, 90]),
        )
        self.assertEqual(result["both_correct"], 1)
        self.assertEqual(result["only_baseline_correct"], 1)
        self.assertEqual(result["only_trained_correct"], 1)
        self.assertEqual(result["neither_correct"], 1)
        self.assertEqual(result["forward_delta_per_example"], -10.0)
        self.assertEqual(result["mcnemar_p_value"], 1.0)

    def test_threshold_comparison_reports_accuracy_and_latency(self):
        baseline = add_latency_metrics(
            {
                "model_label": "base",
                "confidence_threshold": 0.95,
                "examples": 100,
                "correct": 50,
                "accuracy": 0.5,
                "elapsed_seconds": 200.0,
                "total_model_forwards": 1000,
                "tokens_per_forward": 12.8,
            }
        )
        trained = add_latency_metrics(
            {
                "model_label": "trained",
                "confidence_threshold": 0.95,
                "examples": 100,
                "correct": 60,
                "accuracy": 0.6,
                "elapsed_seconds": 100.0,
                "total_model_forwards": 800,
                "tokens_per_forward": 16.0,
            }
        )
        result = comparison(baseline, trained)
        self.assertAlmostEqual(result["accuracy_change_pp"], 10.0)
        self.assertAlmostEqual(result["trained_speedup_vs_baseline"], 2.0)
        self.assertIn("+10.00 percentage points", render_threshold_comparison(result))

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
                json.dumps({"target_source": "frozen_base_confidence_rollout"}),
                encoding="utf-8",
            )
            validate_target_provenance(
                path, [{"target_source": "frozen_base_confidence_rollout"}]
            )
            metadata.write_text(
                json.dumps({"target_source": "frozen_base_local_top1_unlock"}),
                encoding="utf-8",
            )
            validate_target_provenance(
                path, [{"target_source": "frozen_base_local_top1_unlock"}]
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
        self.assertLess(float(correct_loss.grounding), float(wrong_loss.grounding))

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
