# Anchor Lookahead for LLaDA

This directory contains the anchor-lookahead experiment for
`GSAI-ML/LLaDA-8B-Instruct`. The main model is the LoRA adapter selected at
training step 6,000 from the 7,473-example GSM8K run. It learns to expose in one
forward pass token choices that frozen base LLaDA would otherwise reveal over
two consecutive catalyst-decoding steps.

Older IG, Gaussian, anchor-order, rollout, and decoder-search variants are
research history. They are indexed in [experiments/README.md](experiments/README.md).

## Main Result

Full GSM8K test set (`1,319` examples), using the same adaptive decoder for both
models:

| Model | Accuracy | Tokens / forward | Forwards / example | Canvas tokens / second | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen base LLaDA-8B-Instruct | 68.84% | 5.449 | 23.49 | 26.90 | 104.6 min |
| Anchor-lookahead LoRA, step 6,000 | **70.20%** | **7.246** | **17.66** | **33.24** | **84.7 min** |

The accuracy difference is `+1.36` percentage points and the measured canvas
throughput difference is `+23.6%`. The paired accuracy confidence interval
crosses zero, so the accuracy gain should be treated as promising rather than
conclusive. The reduction in model forwards is the stronger result.

See [FINAL_REPORT.md](artifacts/threshold_lookahead_v7/full_train7473_t090_num099/FINAL_REPORT.md)
for the complete metrics and paired comparison.

## Method

### 1. Cache useful canvas states

Start from a blank gold completion canvas. For every plausible alphabetic gold
token, temporarily reveal it at its true position and count how many remaining
positions become both top-1 correct and at least 95% confident under frozen base
LLaDA. Select the token with the largest after-minus-before gain, commit it plus
all correctly unlocked positions, and repeat. A candidate must have at least
70% of the best current gold-token probability. The resulting cache supplies
realistic partial canvases; the 95% threshold is used only to build this cache.

### 2. Distil two teacher steps into one

For each training example, use the blank canvas plus three sampled cached
canvases. With the LoRA disabled, frozen base LLaDA executes two sequential
adaptive catalyst steps at a 90% text threshold and 99% numeric threshold. Each
step chooses the most confident eligible alphabetic token below threshold, or
falls back to the leftmost mask, then commits all threshold-qualified tokens.

The LoRA student sees only the initial canvas. It is optimized with:

```text
loss = future-token CE + target-selection ranking + 5 * base-preservation KL
```

Future-token CE teaches the second teacher action from the earlier canvas. The
ranking loss makes both teacher action positions outrank competing masked
positions. KL preserves frozen-base distributions at other masked positions.
Only `q_proj`, `k_proj`, `v_proj`, and `attn_out` receive rank-8 LoRA updates.

### 3. Decode adaptively

On each forward, select up to two alphabetic below-threshold catalysts. The
second is accepted only when its confidence is at least `0.60` and at least
`0.85` of the first catalyst's confidence. From the same forward, also commit
all text predictions at or above `0.90` and numeric predictions at or above
`0.99`. Repeat until the 128-token canvas is complete.

## Canonical Files

| Stage | File |
| --- | --- |
| Cache generation | `precompute_threshold_unlock_targets.py` |
| Shared catalyst selection | `decode_policy.py` |
| Lookahead trainer | `train_online_lookahead.py` |
| Losses and metrics | `train_lookahead_distillation.py` |
| Matched decoder and evaluation | `eval_threshold_gsm8k.py` |
| Cache command | `run_anchor_lookahead_cache.sh` |
| Exact step-6,000 training command | `run_anchor_lookahead_train.sh` |
| Matched base-versus-trained evaluation | `run_anchor_lookahead_eval.sh` |
| Unit tests | `test_core.py` |

## Reproduce

Run from the repository root inside the project container:

```bash
bash Token2Token/run_anchor_lookahead_cache.sh
bash Token2Token/run_anchor_lookahead_train.sh
bash Token2Token/run_anchor_lookahead_eval.sh
```

The cache and model outputs default to `outputs/token2token/anchor_lookahead/`.
Each script exposes its paths and scale through environment variables at the
top of the file. The original search-and-selection runner remains available as
`run_threshold_lookahead_overnight.sh`, but it is an experiment driver rather
than the canonical final command.
