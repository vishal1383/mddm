# Experimental Variants

The root [README](../README.md) and its three `run_anchor_lookahead_*.sh`
commands define the selected step-6,000 method. Everything else in this folder
is supporting code or an experimental branch retained for reproducibility.

## Experiment Families

| Family | Primary files |
| --- | --- |
| Gaussian IG placement | `train.py`, `run_train.sh` |
| Frozen greedy-IG order | `precompute_anchor_targets.py`, `train_anchor_order.py`, `run_anchor_order.sh` |
| Rollout and local-unlock targets | `precompute_rollout_targets.py`, `precompute_local_unlock_targets.py` |
| Anchor transition V2/V3 | `train_anchor_transition.py`, `run_anchor_transition_v2.sh`, `run_anchor_transition_v3_kl.sh` |
| Parallel unlock V4/V5 | `train_parallel_unlock.py`, `run_parallel_unlock_v4.sh`, `run_threshold_matched_v5.sh` |
| Lookahead V6 searches | `precompute_teacher_rollouts.py`, `train_lookahead_distillation.py`, `run_lookahead_v6.sh` |
| All-unlocked and all-state variants | `train_all_unlocked.py`, `train_all_states_confidence.py` |
| Decoder sweeps and controls | `run_decoder_sweep50*.sh`, `run_full_k123_eval.sh`, `run_standard_confidence_controls.sh` |

The former long-form README is preserved as
[EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md). Consolidated observations are
also available in `../RESEARCH_LOG.md`, `../RESULTS.md`, and the variant-specific
Markdown reports at the package root.

The older Python and shell files intentionally remain at their original paths.
Several recorded commands and runners import those exact module names, so
moving them would break reproducibility without improving the selected method.
