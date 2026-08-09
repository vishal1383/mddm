import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace

import torch

from Token2Token.decode_policy import joint_topk_catalyst_mask
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
    allowed_prediction_mask,
    batch_threshold_unlock_decode,
    batch_topk_decode,
    current_block,
    limit_threshold_selection,
    numeric_prediction_mask,
    parse_thresholds,
    threshold_tag,
    threshold_selection_mask,
)
from Token2Token.precompute_local_unlock_targets import (
    greedy_local_unlock_targets,
    shifted_window,
)
from Token2Token.precompute_rollout_targets import greedy_rollout_targets
from Token2Token.precompute_threshold_unlock_targets import (
    candidate_key,
    correct_threshold_positions,
    decoder_catalyst_position,
    incorrect_threshold_positions,
    inference_aligned_candidates,
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
from Token2Token.commit_phase_analysis import analyse as analyse_commit_phases
from Token2Token.summarize_decoder_sweep import pareto_front
from Token2Token.paired_comparison import (
    compare,
    filtered_shared_ids,
    mcnemar_p_value,
)
from Token2Token.train_parallel_unlock import (
    bucket_positions,
    promoted_fraction,
    gold_cross_entropy,
    preserve_kl,
    promote_objective,
)
from Token2Token.precompute_teacher_rollouts import (
    TARGET_SOURCE as TEACHER_ROLLOUT_SOURCE,
    teacher_rollout_actions,
    validate_rollout_record,
)
from Token2Token.train_lookahead_distillation import (
    future_token_loss,
    sample_rollout_stages,
    target_selection_loss,
)
from Token2Token.train_online_lookahead import (
    normalize_canvas,
    teacher_lookahead_stages,
    threshold_teacher_lookahead_stages,
)
from Token2Token.train_all_unlocked import (
    all_unlocked_stages,
    non_target_kl_loss,
    target_set_cross_entropy,
)
from Token2Token.train_all_states_confidence import (
    chunked,
    select_high_unlock_stages,
    select_target_kind_stages,
    target_confidence_margin_loss,
    target_kind_cross_entropy,
)
from Token2Token.micro_trial_gate import evaluate_candidate
from Token2Token.train_threshold_unlock import trajectory_stages
from Token2Token.summarize_gsm8k_sweep import build_rows, render_report
from Token2Token.select_best_epoch import summarize_epochs
from Token2Token.select_paired_candidate import select_candidate
from Token2Token.select_threshold_operating_point import select_operating_point
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
from Token2Token.analyze_unlock_grid import summarize_grid


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
    def test_joint_catalyst_selection_keeps_top_two_from_one_forward(self):
        confidence = torch.tensor([[0.4, 0.8, 0.7, 0.6]])
        allowed = torch.tensor([[True, True, True, False]])

        chosen = joint_topk_catalyst_mask(confidence, allowed, 2)

        self.assertEqual(chosen.tolist(), [[False, True, True, False]])

    def test_joint_catalyst_selection_gates_only_additional_positions(self):
        confidence = torch.tensor([[0.4, 0.8, 0.69, 0.3]])
        allowed = torch.ones_like(confidence, dtype=torch.bool)

        chosen = joint_topk_catalyst_mask(
            confidence,
            allowed,
            2,
            additional_min_confidence=0.7,
            additional_min_ratio=0.9,
        )

        self.assertEqual(chosen.tolist(), [[False, True, False, False]])

    def test_joint_catalyst_selection_handles_fewer_candidates_than_budget(self):
        confidence = torch.tensor([[0.4, 0.8, 0.7]])
        allowed = torch.tensor([[False, True, False]])

        chosen = joint_topk_catalyst_mask(confidence, allowed, 5)

        self.assertEqual(chosen.tolist(), [[False, True, False]])

    def test_unlock_grid_summary_ranks_safe_causal_gain(self):
        rows = [
            {
                "threshold": 0.8,
                "candidate_prob_ratio": 0.7,
                "wrong_unlock_penalty": 1.0,
                "correct_gain": 3,
                "new_correct": 3,
                "new_wrong": 2,
                "selection_score": 1,
                "gold_probability": 0.8,
                "eligible_candidates": 2,
            },
            {
                "threshold": 0.9,
                "candidate_prob_ratio": 0.7,
                "wrong_unlock_penalty": 1.0,
                "correct_gain": 2,
                "new_correct": 2,
                "new_wrong": 0,
                "selection_score": 2,
                "gold_probability": 0.7,
                "eligible_candidates": 2,
            },
        ]

        summaries = summarize_grid(rows)

        self.assertEqual(summaries[0]["threshold"], 0.9)

    def test_high_unlock_stage_selection_filters_and_ranks_bursts(self):
        stages = [
            {"name": "one", "targets": [{"kind": "catalyst"}, {"kind": "unlocked"}]},
            {
                "name": "three",
                "targets": [{"kind": "catalyst"}] + [{"kind": "unlocked"}] * 3,
            },
            {
                "name": "two",
                "targets": [{"kind": "catalyst"}] + [{"kind": "unlocked"}] * 2,
            },
        ]

        selected = select_high_unlock_stages(
            stages,
            min_unlocked_targets=2,
            min_selection_score=None,
            max_states=2,
        )

        self.assertEqual([stage["name"] for stage in selected], ["three", "two"])

    def test_high_unlock_stage_selection_can_require_correct_catalyst(self):
        stages = [
            {
                "name": "wrong",
                "catalyst_prediction_correct_before": False,
                "targets": [{"kind": "catalyst"}, {"kind": "unlocked"}],
            },
            {
                "name": "correct",
                "catalyst_prediction_correct_before": True,
                "targets": [{"kind": "catalyst"}, {"kind": "unlocked"}],
            },
        ]

        selected = select_high_unlock_stages(
            stages,
            min_unlocked_targets=1,
            catalyst_correct_only=True,
        )

        self.assertEqual([stage["name"] for stage in selected], ["correct"])

    def test_correct_catalyst_filter_rejects_cache_without_provenance(self):
        stages = [
            {
                "name": "legacy",
                "targets": [{"kind": "catalyst"}, {"kind": "unlocked"}],
            }
        ]

        with self.assertRaisesRegex(ValueError, "prediction_correct_before"):
            select_high_unlock_stages(
                stages,
                min_unlocked_targets=1,
                catalyst_correct_only=True,
            )

    def test_micro_trial_gate_uses_paired_quality_and_tpf(self):
        baseline = {
            "1": {"example_id": 1, "correct": True, "model_forwards": 40, "confidence_threshold": 0.8},
            "2": {"example_id": 2, "correct": True, "model_forwards": 40, "confidence_threshold": 0.8},
            "3": {"example_id": 3, "correct": False, "model_forwards": 40, "confidence_threshold": 0.8},
        }
        trained = {
            "1": {"example_id": 1, "correct": True, "model_forwards": 30, "confidence_threshold": 0.8},
            "2": {"example_id": 2, "correct": False, "model_forwards": 30, "confidence_threshold": 0.8},
            "3": {"example_id": 3, "correct": False, "model_forwards": 30, "confidence_threshold": 0.8},
        }

        result = evaluate_candidate(
            baseline,
            trained,
            completion_length=128,
            min_tokens_per_forward=4.0,
            min_tokens_per_forward_gain=0.1,
            max_correct_loss=1,
            expected_threshold=0.8,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["correct_delta"], -1)
        self.assertAlmostEqual(result["trained_tokens_per_forward"], 128 / 30)

    def test_micro_trial_gate_checks_numeric_threshold(self):
        baseline = {
            "1": {
                "example_id": 1,
                "correct": True,
                "model_forwards": 32,
                "confidence_threshold": 0.9,
                "numeric_threshold": 0.99,
            }
        }
        trained = {
            "1": {
                "example_id": 1,
                "correct": True,
                "model_forwards": 32,
                "confidence_threshold": 0.9,
                "numeric_threshold": 0.95,
            }
        }

        with self.assertRaisesRegex(ValueError, "numeric threshold mismatch"):
            evaluate_candidate(
                baseline,
                trained,
                completion_length=128,
                min_tokens_per_forward=4.0,
                min_tokens_per_forward_gain=0.0,
                max_correct_loss=0,
                expected_threshold=0.9,
                expected_numeric_threshold=0.99,
            )

    def test_paired_candidate_selector_prefers_passing_quality_and_speed(self):
        selected, any_passed = select_candidate(
            [
                {
                    "label": "fast_failure",
                    "gate": {
                        "passed": False,
                        "trained_correct": 20,
                        "trained_tokens_per_forward": 8.0,
                    },
                },
                {
                    "label": "safe",
                    "gate": {
                        "passed": True,
                        "trained_correct": 19,
                        "trained_tokens_per_forward": 6.0,
                    },
                },
            ]
        )

        self.assertTrue(any_passed)
        self.assertEqual(selected["label"], "safe")

    def test_teacher_rollout_uses_block_k1_schedule(self):
        canvases, actions = teacher_rollout_actions(
            ToyThresholdModel(),
            [[9]],
            completion_length=3,
            block_length=2,
            mask_token_id=0,
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(canvases, [[1, 2, 3]])
        self.assertEqual(
            [(item["position"], item["token_id"]) for item in actions[0]],
            [(0, 1), (1, 2), (2, 3)],
        )

    def test_lookahead_stages_do_not_cross_block_boundaries(self):
        record = {
            "target_source": TEACHER_ROLLOUT_SOURCE,
            "completion_length": 4,
            "block_length": 2,
            "actions": [
                {"step": index, "position": index, "token_id": index + 1}
                for index in range(4)
            ],
        }
        validate_rollout_record(record)
        stages = sample_rollout_stages(record, 0, 2, 10, random.Random(0))
        self.assertEqual([stage["start"] for stage in stages], [0, 2])
        self.assertEqual(stages[0]["canvas"], [0, 0, 0, 0])
        self.assertEqual(stages[1]["canvas"], [1, 2, 0, 0])

    def test_lookahead_loss_supervises_only_post_anchor_tokens(self):
        logits = torch.zeros(1, 3, 5)
        logits[0, 1, 2] = 10.0
        stage = {
            "canvas": [0, 0, 0],
            "targets": [
                {"position": 0, "token_id": 1},
                {"position": 1, "token_id": 2},
            ],
        }
        loss = future_token_loss(logits, [stage], mask_token_id=0)
        self.assertLess(float(loss), 0.001)

    def test_selection_loss_prefers_all_teacher_positions_over_competitors(self):
        stage = {
            "canvas": [0, 0, 0],
            "targets": [
                {"position": 0, "token_id": 1},
                {"position": 1, "token_id": 2},
            ],
        }
        good = torch.zeros(1, 3, 5)
        good[0, 0, 1] = 8.0
        good[0, 1, 2] = 7.0
        good[0, 2, 3] = 1.0
        bad = good.clone()
        bad[0, 2, 3] = 9.0
        self.assertLess(
            float(target_selection_loss(good, [stage], 0, 3)),
            float(target_selection_loss(bad, [stage], 0, 3)),
        )

    def test_online_teacher_targets_the_post_anchor_action(self):
        stages, _ = teacher_lookahead_stages(
            ToyThresholdModel(),
            [9],
            [[0, 0, 0]],
            mask_token_id=0,
            block_length=3,
            lookahead=2,
            device="cpu",
        )
        self.assertEqual(
            stages[0]["targets"],
            [
                {"position": 0, "token_id": 1},
                {"position": 1, "token_id": 2},
            ],
        )

    def test_threshold_teacher_uses_consecutive_catalyst_actions(self):
        stages, _ = threshold_teacher_lookahead_stages(
            ToyThresholdModel(),
            ToyTextTokenizer(),
            [9],
            [[0, 0, 0]],
            mask_token_id=0,
            lookahead=2,
            confidence_threshold=0.9999,
            numeric_threshold=0.9999,
            device="cpu",
        )
        self.assertEqual(
            stages[0]["targets"],
            [
                {"position": 0, "token_id": 1},
                {"position": 1, "token_id": 2},
            ],
        )

    def test_online_canvas_is_normalized_to_inference_length(self):
        self.assertEqual(normalize_canvas([1, 2], 4, 0), [1, 2, 0, 0])
        self.assertEqual(normalize_canvas([1, 2, 3], 2, 0), [1, 2])

    def test_all_unlocked_stage_targets_catalyst_and_complete_burst(self):
        record = {
            "example_id": "toy",
            "gold_ids": [10, 11, 12, 13, 14],
            "rounds": [
                {
                    "round": 1,
                    "catalyst": {"gold_position": 2, "token_id": 12},
                    "unlocked": [
                        {"gold_position": 0, "token_id": 10},
                        {"gold_position": 3, "token_id": 13},
                    ],
                },
                {
                    "round": 2,
                    "catalyst": {"gold_position": 1, "token_id": 11},
                    "unlocked": [{"gold_position": 4, "token_id": 14}],
                },
            ],
        }
        stages = all_unlocked_stages(record, mask_token_id=0, completion_length=4)
        self.assertEqual(stages[0]["canvas"], [0, 0, 0, 0])
        self.assertEqual(
            [
                (target["position"], target["token_id"], target["kind"])
                for target in stages[0]["targets"]
            ],
            [
                (2, 12, "catalyst"),
                (0, 10, "unlocked"),
                (3, 13, "unlocked"),
            ],
        )
        self.assertEqual(stages[1]["canvas"], [10, 0, 12, 13])
        self.assertEqual(
            [
                (target["position"], target["kind"])
                for target in stages[1]["targets"]
            ],
            [(1, "catalyst")],
        )

    def test_causal_cache_targets_only_new_unlocks_and_corrects_mistakes(self):
        record = {
            "example_id": "causal",
            "gold_ids": [10, 11, 12, 13],
            "rounds": [
                {
                    "round": 1,
                    "selection_score": 2.0,
                    "catalyst": {"gold_position": 2, "token_id": 12},
                    "unlocked": [
                        {"gold_position": 0, "token_id": 10},
                        {"gold_position": 3, "token_id": 13},
                    ],
                    "newly_unlocked": [
                        {"gold_position": 3, "token_id": 13},
                    ],
                    "newly_wrong": [
                        {
                            "gold_position": 1,
                            "gold_token_id": 11,
                            "predicted_token_id": 99,
                        }
                    ],
                }
            ],
        }

        stages = all_unlocked_stages(record, mask_token_id=0, completion_length=4)

        self.assertEqual(stages[0]["selection_score"], 2.0)
        self.assertEqual(
            [
                (target["position"], target["kind"])
                for target in stages[0]["targets"]
            ],
            [(2, "catalyst"), (3, "unlocked"), (1, "correction")],
        )

    def test_all_unlocked_record_without_in_canvas_catalyst_is_skipped(self):
        record = {
            "example_id": "outside",
            "gold_ids": list(range(6)),
            "rounds": [
                {
                    "round": 1,
                    "catalyst": {"gold_position": 5, "token_id": 5},
                    "unlocked": [{"gold_position": 1, "token_id": 1}],
                }
            ],
        }
        self.assertEqual(
            all_unlocked_stages(record, mask_token_id=99, completion_length=4),
            [],
        )

    def test_all_unlocked_ce_averages_each_round_equally(self):
        stages = [
            {
                "canvas": [0, 0, 0],
                "targets": [{"position": 0, "token_id": 1, "kind": "catalyst"}],
            },
            {
                "canvas": [0, 0, 0],
                "targets": [
                    {"position": 0, "token_id": 1, "kind": "catalyst"},
                    {"position": 1, "token_id": 2, "kind": "unlocked"},
                ],
            },
        ]
        logits = torch.zeros(2, 3, 4)
        loss = target_set_cross_entropy(logits, stages, mask_token_id=0)
        self.assertAlmostEqual(float(loss), 1.0986123, places=5)

    def test_all_unlocked_kl_ignores_every_target_position(self):
        stage = {
            "canvas": [0, 0, 0],
            "targets": [
                {"position": 0, "token_id": 1, "kind": "catalyst"},
                {"position": 1, "token_id": 2, "kind": "unlocked"},
            ],
        }
        teacher = torch.zeros(1, 3, 4)
        target_only_change = teacher.clone()
        target_only_change[0, 0, 1] = 8.0
        target_only_change[0, 1, 2] = 8.0
        self.assertAlmostEqual(
            float(non_target_kl_loss(target_only_change, teacher, [stage], 0)),
            0.0,
            places=6,
        )
        non_target_change = target_only_change.clone()
        non_target_change[0, 2, 3] = 8.0
        self.assertGreater(
            float(non_target_kl_loss(non_target_change, teacher, [stage], 0)),
            0.1,
        )

    def test_confidence_margin_is_zero_only_above_probability_floor(self):
        stage = {
            "canvas": [0],
            "targets": [{"position": 0, "token_id": 1, "kind": "catalyst"}],
        }
        high = torch.zeros(1, 1, 4)
        high[0, 0, 1] = 5.0
        low = torch.zeros(1, 1, 4)
        self.assertAlmostEqual(
            float(target_confidence_margin_loss(high, [stage], 0, 0.7)),
            0.0,
            places=6,
        )
        self.assertGreater(
            float(target_confidence_margin_loss(low, [stage], 0, 0.7)),
            0.5,
        )

    def test_confidence_margin_can_target_only_unlocked_tokens(self):
        stage = {
            "canvas": [0, 0],
            "targets": [
                {"position": 0, "token_id": 1, "kind": "catalyst"},
                {"position": 1, "token_id": 2, "kind": "unlocked"},
            ],
        }
        logits = torch.zeros(1, 2, 4)
        logits[0, 0, 1] = -8.0
        logits[0, 1, 2] = 8.0

        loss = target_confidence_margin_loss(
            logits, [stage], 0, 0.7, target_kind="unlocked"
        )

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_selection_stages_can_keep_only_catalysts(self):
        stages = [
            {
                "canvas": [0, 0],
                "targets": [
                    {"position": 0, "token_id": 1, "kind": "catalyst"},
                    {"position": 1, "token_id": 2, "kind": "unlocked"},
                ],
            }
        ]

        selected = select_target_kind_stages(stages, "catalyst")

        self.assertEqual(selected[0]["targets"], [stages[0]["targets"][0]])

    def test_kind_ce_weights_anchor_and_unlocked_targets_separately(self):
        stage = {
            "canvas": [0, 0],
            "targets": [
                {"position": 0, "token_id": 1, "kind": "catalyst"},
                {"position": 1, "token_id": 2, "kind": "unlocked"},
            ],
        }
        logits = torch.zeros(1, 2, 4)
        logits[0, 0, 3] = 8.0
        logits[0, 1, 2] = 8.0
        self.assertLess(
            float(target_kind_cross_entropy(logits, [stage], 0, "unlocked")),
            0.001,
        )
        self.assertGreater(
            float(target_kind_cross_entropy(logits, [stage], 0, "catalyst")),
            7.0,
        )

    def test_state_chunks_cover_every_state_once(self):
        states = list(range(11))
        batches = chunked(states, 4)
        self.assertEqual([len(batch) for batch in batches], [4, 4, 3])
        self.assertEqual([item for batch in batches for item in batch], states)

    def test_threshold_selector_requires_quality_and_four_token_speed(self):
        baseline = {"accuracy": 0.75, "tokens_per_forward": 3.2}
        candidates = [
            {"confidence_threshold": 0.9, "accuracy": 0.75, "tokens_per_forward": 3.9},
            {"confidence_threshold": 0.8, "accuracy": 0.70, "tokens_per_forward": 5.0},
            {"confidence_threshold": 0.7, "accuracy": 0.74, "tokens_per_forward": 4.2},
        ]
        selected = select_operating_point(
            baseline,
            candidates,
            accuracy_tolerance=0.02,
            minimum_tokens_per_forward=4.0,
        )
        self.assertEqual(selected["confidence_threshold"], 0.7)

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

    def test_numeric_predictions_use_the_higher_threshold(self):
        class MixedTokenizer(ToyTokenizer):
            def decode(self, token_ids):
                return {1: " total", 2: " 42", 3: "3rd"}[token_ids[0]]

        eligible = torch.tensor([[True, True, True]])
        confidence = torch.tensor([[0.91, 0.95, 0.99]])
        token_ids = torch.tensor([[1, 2, 3]])
        numeric = numeric_prediction_mask(
            token_ids, eligible, MixedTokenizer(), {}
        )
        selected = threshold_selection_mask(
            eligible,
            confidence,
            token_ids,
            0.9,
            0.98,
            MixedTokenizer(),
            {},
        )
        self.assertEqual(numeric.tolist(), [[False, True, True]])
        self.assertEqual(selected.tolist(), [[True, False, True]])

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

    def test_when_empty_skips_the_forced_commit_if_threshold_selects(self):
        canvases, stats = batch_threshold_unlock_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            confidence_threshold=0.95,
            commit_threshold_on_first_forward=True,
            unlock_forward=False,
            force_catalyst="when-empty",
            tokenizer=ToyTextTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(canvases, [[1, 2, 3]])
        # Positions 0 and 1 clear the threshold, so nothing is forced on the
        # first forward; only the last position needs a forced commit.
        self.assertEqual(stats[0]["catalyst_tokens"], 1)
        self.assertEqual(stats[0]["first_forward_threshold_tokens"], 2)
        self.assertEqual(stats[0]["model_forwards"], 2)

    def test_when_empty_requires_first_forward_threshold_commits(self):
        with self.assertRaises(ValueError):
            batch_threshold_unlock_decode(
                ToyThresholdModel(),
                [[9]],
                3,
                0,
                confidence_threshold=0.95,
                force_catalyst="when-empty",
                tokenizer=ToyTextTokenizer(),
                device="cpu",
                pad_token_id=5,
            )

    def test_catalyst_min_length_excludes_short_words(self):
        class ShortWordTokenizer(ToyTokenizer):
            def decode(self, token_ids):
                return {1: " is", 2: " apples", 3: " the"}.get(token_ids[0], "x")

        masked = torch.tensor([[True, True, True]])
        tokens = torch.tensor([[1, 2, 3]])
        unfiltered = allowed_prediction_mask(
            tokens, masked, ShortWordTokenizer(), {}, "text", 0
        )
        self.assertEqual(unfiltered[0].tolist(), [True, True, True])
        # "is" and "the" are high-confidence but carry no information, which is
        # the same failure mode that sank the any-token filter.
        filtered = allowed_prediction_mask(
            tokens, masked, ShortWordTokenizer(), {}, "text", 4
        )
        self.assertEqual(filtered[0].tolist(), [False, True, False])

    def test_below_filter_forces_only_tokens_the_threshold_would_skip(self):
        # ToyThresholdModel: position 0 and 1 clear 0.95, position 2 does not.
        canvases, stats = batch_threshold_unlock_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            confidence_threshold=0.95,
            commit_threshold_on_first_forward=True,
            unlock_forward=False,
            catalyst_filter="below",
            tokenizer=ToyTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(canvases, [[1, 2, 3]])
        # The one forced commit goes to position 2, the only position the
        # threshold would not have taken, so the whole canvas lands in one
        # forward instead of two.
        self.assertEqual(stats[0]["model_forwards"], 1)
        self.assertEqual(stats[0]["catalyst_tokens"], 1)
        self.assertEqual(stats[0]["first_forward_threshold_tokens"], 2)

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

    def test_high_threshold_any_decoder_reduces_to_block_k1(self):
        adaptive, adaptive_stats = batch_threshold_unlock_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            confidence_threshold=0.999999,
            commit_threshold_on_first_forward=True,
            unlock_forward=False,
            catalyst_filter="any",
            block_length=2,
            tokenizer=ToyTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        standard, standard_stats = batch_topk_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            tokens_per_step=1,
            block_length=2,
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(adaptive, standard)
        self.assertEqual(adaptive_stats[0]["model_forwards"], 3)
        self.assertEqual(standard_stats[0]["model_forwards"], 3)

    def test_gated_adapter_requires_an_unlock_forward_to_gate(self):
        with self.assertRaises(ValueError):
            batch_threshold_unlock_decode(
                ToyThresholdModel(),
                [[9]],
                3,
                0,
                confidence_threshold=0.95,
                commit_threshold_on_first_forward=True,
                unlock_forward=False,
                base_first_forward=True,
                tokenizer=ToyTextTokenizer(),
                device="cpu",
                pad_token_id=5,
            )

    def test_commit_phase_records_which_rule_placed_each_token(self):
        canvases, stats = batch_threshold_unlock_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            confidence_threshold=0.95,
            record_commit_phase=True,
            tokenizer=ToyTextTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(canvases, [[1, 2, 3]])
        phases = stats[0]["commit_phase"]
        self.assertEqual(len(phases), 3)
        self.assertEqual(phases[0], 1)
        # Position 1 clears the threshold only after the catalyst is placed,
        # so it is an unlock commit rather than a catalyst commit.
        self.assertEqual(phases[1], 4)
        self.assertTrue(all(phase in (1, 2, 3, 4) for phase in phases))

    def test_commit_phase_is_omitted_unless_requested(self):
        _, stats = batch_threshold_unlock_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            confidence_threshold=0.95,
            tokenizer=ToyTextTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        self.assertNotIn("commit_phase", stats[0])

    def test_block_length_confines_anchor_and_burst_to_one_block(self):
        # ToyThresholdModel is confident at positions 0 and 1, not 2. With a
        # block of 1, the burst cannot reach position 1 until the block moves,
        # so every position costs its own forward.
        canvases, stats = batch_threshold_unlock_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            confidence_threshold=0.95,
            commit_threshold_on_first_forward=True,
            unlock_forward=False,
            block_length=1,
            tokenizer=ToyTextTokenizer(),
            device="cpu",
            pad_token_id=5,
        )
        self.assertEqual(canvases, [[1, 2, 3]])
        self.assertEqual(stats[0]["model_forwards"], 3)
        self.assertEqual(stats[0]["first_forward_threshold_tokens"], 0)

    def test_unbounded_block_lets_the_burst_run_ahead(self):
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
        self.assertEqual(stats[0]["model_forwards"], 2)
        self.assertEqual(stats[0]["first_forward_threshold_tokens"], 1)

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

    def test_current_block_tracks_the_leftmost_unfinished_block(self):
        masked = torch.tensor([[False, False, True, True], [True, True, True, True]])
        blocks = current_block(masked, 2)
        # First row is filled through position 1, so the active block is [2,4).
        self.assertEqual(blocks[0].tolist(), [False, False, True, True])
        self.assertEqual(blocks[1].tolist(), [True, True, False, False])

    def test_block_decoding_fills_left_to_right(self):
        canvases, stats = batch_topk_decode(
            ToyThresholdModel(),
            [[9]],
            3,
            0,
            tokens_per_step=1,
            block_length=1,
            device="cpu",
            pad_token_id=5,
        )
        # Position 2 is the model's least confident, but block decoding still
        # reaches it, which global confidence ordering would defer.
        self.assertEqual(canvases, [[1, 2, 3]])
        self.assertEqual(stats[0]["model_forwards"], 3)

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

    def test_threshold_mistake_set_keeps_only_wrong_high_confidence_predictions(self):
        mistakes = incorrect_threshold_positions(
            [0, 0, 7, 0, 0],
            [1, 2, 7, 4, 5],
            [1, 9, 7, 4, 8],
            [0.96, 0.99, 0.99, 0.951, 0.949],
            candidate_position=0,
            mask_token_id=0,
            threshold=0.95,
        )
        self.assertEqual([item["gold_position"] for item in mistakes], [1])

    def test_mixed_threshold_requires_higher_confidence_for_numbers(self):
        class NumericTokenizer(ToyTextTokenizer):
            def decode(self, token_ids):
                return {1: "one", 2: "2", 3: "three"}[token_ids[0]]

        unlocked = correct_threshold_positions(
            [0, 0, 0],
            [1, 2, 3],
            [1, 2, 3],
            [0.95, 0.95, 0.95],
            candidate_position=0,
            mask_token_id=0,
            threshold=0.9,
            tokenizer=NumericTokenizer(),
            numeric_threshold=0.99,
        )
        self.assertEqual([item["gold_position"] for item in unlocked], [2])

    def test_decoder_catalyst_matches_highest_confidence_text_below(self):
        position, eligible, cleanup = decoder_catalyst_position(
            [0, 1, 2],
            prediction=[1, 2, 3],
            confidence=[0.4, 0.8, 0.7],
            tokenizer=ToyTextTokenizer(),
            threshold=0.9,
        )
        self.assertEqual((position, eligible, cleanup), (1, 3, False))

    def test_decoder_catalyst_falls_back_to_leftmost_cleanup(self):
        position, eligible, cleanup = decoder_catalyst_position(
            [1, 2],
            prediction=[1, 2, 3],
            confidence=[0.99, 0.99, 0.99],
            tokenizer=ToyTextTokenizer(),
            threshold=0.9,
        )
        self.assertEqual((position, eligible, cleanup), (1, 0, True))

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
        # This toy already predicts position 1 above threshold before the
        # catalyst, so it is committed but is not a causal new unlock.
        self.assertEqual(rounds[0]["newly_unlocked"], [])
        self.assertEqual(rounds[0]["new_correct_unlocks"], 0)
        self.assertEqual(rounds[1]["catalyst"]["gold_position"], 2)

    def test_decoder_target_trajectory_uses_decoder_position(self):
        rounds, residual = threshold_unlock_trajectory(
            ToyThresholdModel(),
            ToyTextTokenizer(),
            [9],
            [1, 2, 3],
            0,
            confidence_threshold=0.95,
            candidate_prob_ratio=0.5,
            candidate_batch_size=2,
            selection_mode="decoder",
            device="cpu",
        )
        self.assertEqual(residual, [])
        self.assertEqual(rounds[0]["catalyst"]["gold_position"], 2)
        self.assertFalse(rounds[0]["catalyst"]["decoder_cleanup"])

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

    def test_inference_aligned_candidates_require_correct_below_tau(self):
        self.assertEqual(
            inference_aligned_candidates(
                [0, 1, 2],
                prediction=[9, 2, 3],
                confidence=[0.4, 0.95, 0.7],
                gold_ids=[1, 2, 3],
                threshold=0.9,
                require_correct=True,
                require_below_threshold=True,
            ),
            [2],
        )

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
            float(promote_objective(committable, [[0]], [1, 2], zero, "hinge", 0.97, 0)),
            0.0,
        )
        self.assertGreater(
            float(promote_objective(committable, [[0]], [1, 2], zero, "ce", 0.97, 0)),
            0.0,
        )
        undecided = torch.zeros(1, 2, 6)
        undecided[0, 0, 1] = 1.8
        self.assertGreater(
            float(promote_objective(undecided, [[0]], [1, 2], zero, "hinge", 0.97, 0)),
            0.0,
        )

    def test_promoted_fraction_counts_positions_past_the_threshold(self):
        logits = torch.zeros(1, 2, 6)
        logits[0, 0, 1] = 12.0
        logits[0, 1, 2] = 1.0
        self.assertEqual(promoted_fraction(logits, [[0]], [1, 2], 0.95, 0), 1.0)
        self.assertEqual(promoted_fraction(logits, [[1]], [1, 2], 0.95, 0), 0.0)
        self.assertEqual(promoted_fraction(logits, [[0, 1]], [1, 2], 0.95, 0), 0.5)
        self.assertEqual(promoted_fraction(logits, [[]], [1, 2], 0.95, 0), 0.0)

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

    def test_commit_phase_analysis_contrasts_correct_and_wrong(self):
        rows = [
            {"correct": True, "commit_phase": [1, 1, 3, 3]},
            {"correct": True, "commit_phase": [1, 3, 3, 3]},
            {"correct": False, "commit_phase": [4, 4, 4, 1]},
            {"correct": False, "commit_phase": [4, 4, 1, 1]},
        ]
        report = analyse_commit_phases(rows)
        self.assertEqual(report["examples"], 4)
        self.assertEqual(report["correct"], 2)
        unlock = next(
            item
            for item in report["comparisons"]
            if item["phase"] == "post-anchor unlock"
        )
        self.assertEqual(unlock["correct_share"], 0.0)
        self.assertGreater(unlock["wrong_share"], 0.0)
        self.assertGreater(unlock["difference"], 0.0)

    def test_pareto_front_keeps_only_undominated_configs(self):
        def row(name, accuracy, tokens_per_forward):
            return {
                "config": name,
                "accuracy": accuracy,
                "tokens_per_forward": tokens_per_forward,
            }

        rows = [
            row("single_forward", 0.72, 3.089),
            row("catalyst_uncapped", 0.68, 2.065),
            row("fast_but_poor", 0.50, 5.0),
            row("slow_but_good", 0.75, 1.0),
        ]
        front = {item["config"] for item in pareto_front(rows)}
        # Dominated on both axes by single_forward.
        self.assertNotIn("catalyst_uncapped", front)
        # Each of these wins on one axis, so none of them is dominated.
        self.assertEqual(front, {"single_forward", "fast_but_poor", "slow_but_good"})

    def test_pareto_front_drops_exact_duplicates_of_a_better_row(self):
        rows = [
            {"config": "a", "accuracy": 0.7, "tokens_per_forward": 3.0},
            {"config": "b", "accuracy": 0.7, "tokens_per_forward": 2.0},
        ]
        front = {item["config"] for item in pareto_front(rows)}
        self.assertEqual(front, {"a"})

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

    def test_paired_comparison_can_hold_out_an_id_range(self):
        baseline = {str(i): {} for i in range(6)}
        trained = {str(i): {} for i in range(1, 7)}
        self.assertEqual(
            filtered_shared_ids(
                baseline, trained, min_example_id=2, max_example_id=5
            ),
            ["2", "3", "4"],
        )

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
