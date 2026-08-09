# Experimental Variants

The root [README](../README.md), the `../main/` package, and the three launchers
in `../scripts/` define the selected step-6,000 method. Everything in this
folder is an experimental branch retained for reproducibility.

## Experiment Families

| Family | Primary files |
| --- | --- |
| Gaussian IG placement | `../main/train.py`, `scripts/run_train.sh` |
| Frozen greedy-IG order | `../main/precompute_anchor_targets.py`, `train_anchor_order.py`, `scripts/run_anchor_order.sh` |
| Rollout and local-unlock targets | `precompute_rollout_targets.py`, `precompute_local_unlock_targets.py` |
| Anchor transition V2/V3 | `../main/train_anchor_transition.py`, `scripts/run_anchor_transition_v2.sh`, `scripts/run_anchor_transition_v3_kl.sh` |
| Parallel unlock V4/V5 | `train_parallel_unlock.py`, `scripts/run_parallel_unlock_v4.sh`, `scripts/run_threshold_matched_v5.sh` |
| Lookahead V6 searches | `../main/precompute_teacher_rollouts.py`, `../main/train_lookahead_distillation.py`, `scripts/run_lookahead_v6.sh` |
| All-unlocked and all-state variants | `train_all_unlocked.py`, `train_all_states_confidence.py` |
| Decoder sweeps and controls | `scripts/run_decoder_sweep50*.sh`, `scripts/run_full_k123_eval.sh`, `scripts/run_standard_confidence_controls.sh` |

The former long-form README is preserved as
[EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md). Consolidated observations
and variant-specific reports are under `docs/`.

Archived Python modules remain importable as `Token2Token.experiments.*`, and
their launchers remain runnable from `Token2Token/experiments/scripts/`.
