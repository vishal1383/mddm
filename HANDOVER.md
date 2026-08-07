# MDDM Anchor / Token2Token Project Handover

Last updated: 2026-08-07

## 1. Current State

- Repository: `https://github.com/vishal1383/mddm`
- Active branch: `agent/token2token-anchor-training`
- Current implementation commit: `14e57e4`
  (`Preserve base behavior during anchor training`)
- Base model throughout the latest work: `GSAI-ML/LLaDA-8B-Instruct`
- Main dataset: GSM8K (`openai/gsm8k`, `main`)
- Persistent Docker container: `confident_borg`
- Container project directory: `/workspace/DhruveshProject`
- Host project directory: `/home/vishalg/Desktop/DhruveshProject`
- No training or evaluation process is running.
- No tmux session is running.
- The attempted V3 early-100 run was stopped before step 1. It produced only
  `outputs/token2token/anchor_transition_v3/train100_kl5/config.json`; there is
  no useful checkpoint to resume.

The working tree contains unrelated generated `__pycache__` changes, report
outputs, and `PapersMisc/RecentMDDMpaper.pdf`. Do not revert or stage these by
accident. All intended Token2Token source changes through V3 are committed and
pushed.

## 2. Original Anchor-Decode Experiment

### Question

The initial question was whether cheated/gold anchors chosen using greedy
information gain (IG) improve the final decoded GSM8K answer, and why accuracy
eventually falls when more anchors are added.

For every example and every `k`:

1. Reset to the original masked completion canvas.
2. Place the first `k` gold tokens selected by the anchor policy.
3. Decode all remaining positions using the same standard confidence-greedy
   LLaDA decoder.
4. Extract and score the final GSM8K answer.

The prefix control places the first `k` anchorable gold completion tokens from
left to right. Every `k` is a fresh decode. `k=10` is not equivalent to a full
ordinary decode; it means ten gold positions are fixed and the rest are
decoded.

### Full GSM8K result (1,319 test examples)

| k | Greedy-IG accuracy | Delta vs k=0 | Prefix accuracy | Delta vs k=0 |
|---:|---:|---:|---:|---:|
| 0 | 54.89% | +0.00 pp | 54.89% | +0.00 pp |
| 1 | 66.72% | +11.83 pp | 54.81% | -0.08 pp |
| 2 | 68.76% | +13.87 pp | 56.48% | +1.59 pp |
| 3 | 68.01% | +13.12 pp | 55.88% | +0.99 pp |
| 4 | 68.39% | +13.50 pp | 57.47% | +2.58 pp |
| 5 | 65.73% | +10.84 pp | 59.59% | +4.70 pp |
| 6 | 64.52% | +9.63 pp | 57.39% | +2.50 pp |
| 7 | 62.93% | +8.04 pp | 59.21% | +4.32 pp |
| 8 | 61.26% | +6.37 pp | 60.50% | +5.61 pp |
| 9 | 59.44% | +4.55 pp | 63.00% | +8.11 pp |
| 10 | 59.14% | +4.25 pp | 62.62% | +7.73 pp |

The important broad result is that greedy IG gives a large early gain, peaks
around `k=2`, and then declines. Local prefix rises and falls are paired-sample
variation; 0/10 adjacent prefix changes survive Bonferroni correction.

### Main failure pattern

From greedy `k=2` to `k=10`:

- Stay correct: 686
- Regress: 221
- Recover: 94
- Stay wrong: 318
- Net: -127 examples

First harmful anchor types in the 221 regressions:

- Number: 144
- Word: 34
- Operator: 18
- Punctuation: 10
- Final marker: 8
- Calculation marker: 7
- Late first harmful anchor: 142/221 (64.3%)

The strongest concrete hypothesis is the stray/partial-number-anchor problem,
especially for late digits. Example 349 is the clearest case: at `k=9` the
decoded answer is `2640`; adding one gold final-answer digit `0` at position
110 causes the free decoder to produce `26400`.

This is a supported aggregate pattern, not proof that every regression is
caused by a stray digit. Other likely causes are late fragments, objective
mismatch (token confidence rather than answer correctness), and conflict
between the gold rationale and the model's preferred solution path.

### Anchor-analysis artifacts

Primary report:

- `decode_impact_analysis/llada-8b_gsm8k/completion_hypothesis_report.docx`
- Markdown source:
  `decode_impact_analysis/llada-8b_gsm8k/completion_hypothesis_report.md`
- Google Doc shared with Dhruvesh:
  `https://docs.google.com/document/d/1hSPDjkc4Oaxb_YsDJtOs1QaK3NnBBxa7EjuQdhBF85o/edit?usp=sharing`

Raw full-run data and plots:

- `outputs/decode_impact_all/llada-8b_gsm8k/`
- `outputs/decode_impact_full/llada-8b_gsm8k/`
- Important files include:
  - `anchor_decode_trajectories.jsonl`
  - `anchor_decode_tokens.jsonl`
  - `anchor_answer_timeline.jsonl`
  - `anchor_token_effects.jsonl`
  - `greedy_standard_accuracy_change_by_example.jsonl`
  - `plots/greedy_vs_left_to_right_accuracy_change.png`

The polished report is the source of truth for the full 1,319-example table.
The older `outputs/decode_impact_full/.../decode_impact_report.md` is only the
earlier 20-example run.

## 3. Research Goal That Followed

The desired training idea became:

- Avoid a separate RL policy/anchor network.
- Use tokens selected from a frozen base model as supervision.
- Train LLaDA to place useful anchor/catalyst tokens before ordinary tokens.
- After an anchor is placed, make multiple other positions confidently and
  correctly decodable.
- Ultimately beat or match base LLaDA quality while reducing decoding model
  forwards, especially relative to confidence decoding with `k=2`.
- Full Token2Token correction/replacement of wrong committed tokens is a later
  phase, not implemented yet.

This was discussed in relation to Anchored Diffusion Language Models, but the
intended distinction is that anchor discovery comes from frozen-base IG or
unlock statistics rather than a learned policy network.

### Communication already sent to Dhruvesh

Dhruvesh was sent the Google Doc above and told that the full GSM8K
LLaDA-8B-Instruct experiment improves with a few greedy-IG anchors and then
drops, with the leading hypothesis being stray late numeric fragments such as
`2640 -> 26400`.

He was also sent the proposed first training step: use frozen greedy-IG gold
tokens as anchor supervision with anchor-placement/relative-order losses,
evaluate normal decoding, and defer full Token2Token replacement of wrong
tokens until that first stage works. The subsequent experiments in this
handover show that this initial training formulation did not work, so any new
update should explicitly mention the negative results rather than implying the
trainer is already successful.

## 4. Token2Token Implementation Timeline

All current source lives under `Token2Token/` and is intentionally independent
of `mdm_probe`.

### 4.1 Initial Gaussian / relative-order trainer

The first trainer followed the supplied Gaussian-proposal image at a high
level: anchor placement, relative-order preservation, and CE, using LoRA rather
than full fine-tuning. It exposed several issues:

- Relative-order loss was effectively near zero in logs.
- Some selected anchors were symbols and partial numeric tokens.
- Online anchor generation changed as the LoRA changed, which violated the
  intended fixed-gold supervision.
- A Transformers/LLaDA compatibility issue was fixed where
  `tie_weights(missing_keys=...)` was incompatible with the remote model.

### 4.2 Frozen greedy-IG anchor-order training

Anchor lists were changed to be precomputed once from base LLaDA and then kept
fixed. Standard denoising/sequence CE was added after the anchor sequence.

This approach substantially damaged ordinary decoding:

- Matched first 50 examples: base 56%, standard denoising LoRA 60%, anchor
  model 36%.
- Full GSM8K epoch checkpoints using `k=1` decode:
  - Epoch 1: 449/1319 = 34.04%
  - Epoch 2: 427/1319 = 32.37%
  - Epoch 5/best train loss: 491/1319 = 37.23%
- Epoch-5 accuracy by tokens per step:
  - `k=1`: 37.23%
  - `k=2`: 34.72%
  - `k=3`: 28.51%

Artifacts are in the container under:

- `outputs/token2token/anchor_order_plus_completion_gsm8k_5epoch_run1/`
- `outputs/token2token/eval_anchor_epochs_k1_full/`
- `outputs/token2token/eval_anchor_order_5epoch_best_train_loss/`

Do not continue this trainer for more epochs; degradation was already clear.

### 4.3 Alternative frozen target metrics

Several target selectors were implemented while searching for a target that
directly predicts decoding speedup:

- `precompute_anchor_targets.py`: frozen greedy IG order
- `precompute_rollout_targets.py`: choose gold anchors by correct rollout count
- `precompute_local_unlock_targets.py`: local/windowed top-1 unlock count
- `precompute_threshold_unlock_targets.py`: full-canvas confidence-threshold
  gain, which became the main target cache

Related commits:

- `926cc8b` frozen-IG experiments
- `06cb459` rollout selection
- `4b63612` local top-1 unlock selection
- `bdfa796` threshold-unlock targets/training
- `d201094` after-minus-before gain

## 5. Frozen Threshold-Gain Target Cache

This is the expensive reusable target asset. It is complete; do not regenerate
it unless the target definition changes.

Container path:

`outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl`

Metadata:

- 7,473 GSM8K training examples
- 918,080 completion tokens
- Maximum completion length 512; all GSM8K gold completions fit
- Full completion canvas, not a 128-token training truncation
- Candidate filter: alphabetic decoded tokens only
- Reject whitespace, numbers, punctuation, operators, markup, and tokenizer
  special tokens as catalysts
- Candidate plausibility: gold probability at least `0.7 * max gold probability`
- For every candidate, compare the number of correctly placed top-1 gold
  tokens with confidence at least 0.95 before and after placing it
- Select by largest `after - before`; tie by `after`, then candidate probability
- Original target generation teacher-forced the selected catalyst and every
  correct >=0.95 unlocked token, then repeated

Cache statistics:

- Rounds: 185,367
- Unlocked tokens: 384,600
- Mean correct-95 gain: 1.4162
- Mean unlocked per round: 2.0748
- Mean tokens placed per round: 3.0748
- Zero-unlock rounds: 46.40%
- Residual tokens: 348,113
- Implied tokens per forward with original cleanup accounting: 1.2772

Files:

- Cache: path above (about 142 MB)
- Summary: same basename with `.summary.json`
- Config: same basename with `.config.json`

Important provenance string checked by trainers:
`frozen_base_threshold_gain_text_anchors`.

## 6. Threshold Anchor-Only + LTR Cleanup Run

The strict trainer initially applied CE only to selected text anchors. When no
eligible text anchors remained, residual positions were changed to single-token
left-to-right cleanup stages. Inference used the same text-only catalyst filter,
then an unrestricted >=0.95 burst, and left-to-right residual cleanup.

Full training completed:

- 7,473/7,473 records
- Final adapter saved
- Training time: 10,737 seconds
- Adapter:
  `outputs/token2token/threshold_unlock/llada8b_gsm8k_t095_text_q07_anchor_ltr_1epoch/adapter-final`

Full base evaluation under the same threshold decoder:

- 951/1319 = 72.10%
- Total forwards: 75,647
- Tokens per forward: 2.2318

Trained evaluation was interrupted at 816 examples but was already decisive:

- 293/816 = 35.91%

Main diagnosis:

1. The trainer learned `canvas -> anchor A` with CE.
2. It then inserted anchor A and unlocked tokens B/C as gold.
3. It never applied CE to learn `canvas + A -> B/C` while B/C were masked.
4. Therefore it did not train the causal unlock behavior at all.
5. LTR residual stages dominated many sampled batches and distorted normal
   denoising.
6. Standard denoising preservation was zero in this strict run.
7. Fine-tuning shifted confidence calibration, while inference irreversibly
   committed predictions at the old 0.95 threshold.

This run should not be resumed.

## 7. Capped Anchor-Transition V2

V2 corrected the missing transition loss:

1. Train the selected frozen-base anchor before placement.
2. Place only the gold anchor.
3. Run the current model on the post-anchor canvas.
4. Select its current top two masked positions by confidence, whether correct
   or wrong.
5. Apply gold CE at those two positions.
6. Include one ordinary random-denoising CE canvas per update.
7. Do not train residual LTR stages.

Matching inference places one eligible text catalyst, reruns the model, and
commits at most two >=0.95 positions. Both base and trained models use exactly
the same decoder.

V2 configuration at checkpoint 500:

- 500 examples used before stopping
- Learning rate `1e-5`
- LoRA rank 8, alpha 16
- Targets: `q_proj,k_proj,v_proj,attn_out`
- Loss weights: standard 1.0, anchor 0.5, post-anchor 1.0
- Three sampled transitions/example
- Two dynamic post-anchor targets/transition

Matched 50-example GSM8K test result:

| Model | Accuracy | Forwards/example | Seconds/example | Tokens/forward |
|---|---:|---:|---:|---:|
| Base LLaDA | 34/50 = 68% | 103.70 | 14.87 | 1.234 |
| V2 step 500 | 22/50 = 44% | 119.58 | 21.27 | 1.070 |

V2 failed quality and latency. It reduced mean threshold commits from 65.62 to
38.08, showing severe confidence-calibration damage.

Artifacts:

- Checkpoint:
  `outputs/token2token/anchor_transition_v2/full_lr1e5_top2/checkpoint-000500`
- Comparison:
  `outputs/token2token/anchor_transition_v2/eval50_step500_lr1e5_top2/comparison.md`

## 8. KL-Preserved V3

V3 added frozen-base behavior preservation without loading a second 8B model.
The same PEFT model is run with `model.disable_adapter()` under `no_grad` to
produce teacher logits on the random denoising canvas. KL is applied between
the base and adapted distributions at masked positions.

V3 configuration:

- 500 training examples
- Learning rate `5e-6`
- LoRA rank 4, alpha 8
- Only `v_proj` and `attn_out`
- 2,097,152 trainable parameters
- Standard CE weight 1.0
- Anchor CE weight 0.25
- Dynamic post-anchor CE weight 0.5
- Base KL weight 5.0
- Three transitions/example, two dynamic post-anchor targets each

Matched 50-example result:

| Model | Accuracy | Forwards/example | Seconds/example | Tokens/forward |
|---|---:|---:|---:|---:|
| Base LLaDA | 34/50 = 68% | 103.70 | 14.87 | 1.234 |
| V3 KL-5 | 32/50 = 64% | 106.34 | 15.57 | 1.204 |

V3 recovered most of V2's damage but did not meet the success criterion. In
paired outcomes it gained two answers and lost four relative to base. It also
used 132 more total forwards across the 50 examples.

Artifacts:

- Adapter:
  `outputs/token2token/anchor_transition_v3/train500_kl5/adapter-final`
- Comparison:
  `outputs/token2token/anchor_transition_v3/eval50_kl5/comparison.md`
- Training log:
  `outputs/token2token/anchor_transition_v3/train500_kl5/train.jsonl`

The attempted V3 early-100 rerun was stopped before useful work and can be
deleted or ignored.

## 9. Current Success Criterion

On the exact same 50 GSM8K test examples and identical capped confidence
decoder, a candidate passes only if:

1. Accuracy is at least the base accuracy (currently at least 34/50).
2. Total model forwards are lower than base (currently below 5,185).
3. Wall seconds/example are lower than base (currently below 14.87).

This is only a smoke criterion. Any passing configuration should then be run on
all 1,319 GSM8K test examples with paired confidence intervals.

Also retain the original broader goal: compare against standard fixed `k=2`
confidence decoding. The recent V2/V3 benchmark compares base and trained under
the custom capped catalyst decoder, not yet against the ordinary fixed-`k=2`
baseline.

## 10. Important Compute Constraint

The current catalyst algorithm uses two forwards per cycle:

1. One forward to choose/place a catalyst.
2. One forward after placement to identify the confidence burst.

With a cap of two burst tokens, a cycle places at most three tokens over two
forwards: at most 1.5 tokens/forward. Therefore this capped diagnostic cannot
beat a perfect ordinary `k=2` decoder in raw forward count. It is useful for
checking whether the causal anchor mechanism can work without catastrophic
errors. A final speed-oriented design must eventually do at least one of:

- Unlock at least four additional tokens after one catalyst (five tokens over
  two forwards beats two tokens/forward).
- Place multiple catalysts in the first forward.
- Remove the dedicated second forward.
- Use speculative/parallel verification.

## 11. Recommended Next Experiments

Priority order:

1. **Gated adapter experiment.** Use base LLaDA (adapter disabled) for catalyst
   selection and ordinary steps. Enable the adapter only for the post-anchor
   forward. Train only the post-anchor transition. This isolates the learned
   behavior and prevents the adapter from globally changing anchor selection
   and confidence calibration.
2. **Stronger distillation with early checkpoints.** Train 50/100/200 examples,
   save every 50, and try KL weights 20-100. V3 KL-5 was close but still moved
   the distribution. Do not commit to 500/full epochs before evaluating early
   checkpoints.
3. **Explicit confidence/ranking supervision.** Anchor CE does not directly
   teach the selected anchor's confidence to outrank confidently wrong
   positions. Add a margin that ranks the correct anchor above incorrect top-1
   candidates, while not suppressing positions whose top-1 token is already
   gold-correct.
4. **Threshold calibration sweep.** Compare base and trained at identical
   thresholds such as 0.90, 0.95, 0.99. Fine-tuning changes calibration, so a
   fixed 0.95 is not automatically comparable. Report the accuracy/forward
   Pareto frontier, not a cherry-picked single threshold.
5. **Use current-model top-two mistakes plus cached causal targets carefully.**
   V2/V3 dynamically train current top-two positions, including mistakes. The
   cached correct >=0.95 positions can be added as low-weight positive targets
   if the goal is to increase burst density rather than only repair mistakes.
6. **Numeric span handling.** For any future numeric catalysts, never place one
   isolated digit. Either ban numeric catalysts, as current text-only targets
   do, or treat a full contiguous number as one grouped action.
7. **Only after a smoke win:** train all 7,473 examples, evaluate all 1,319 test
   examples, then repeat on LM1B with likelihood/perplexity-style quality and
   decoding forward count.

## 12. Key Source Files

- `Token2Token/README.md`: runnable overview
- `Token2Token/train.py`: model loading, LoRA setup, compatibility patches
- `Token2Token/core.py`: original anchor/position losses
- `Token2Token/precompute_anchor_targets.py`: frozen IG targets
- `Token2Token/precompute_rollout_targets.py`: rollout selector
- `Token2Token/precompute_local_unlock_targets.py`: local unlock selector
- `Token2Token/precompute_threshold_unlock_targets.py`: current target cache
- `Token2Token/train_anchor_order.py`: frozen anchor-order trainer
- `Token2Token/train_threshold_unlock.py`: strict threshold/LTR trainer
- `Token2Token/train_anchor_transition.py`: V2/V3 transition trainer
- `Token2Token/decode.py`: single-example confidence/catalyst decoders
- `Token2Token/eval_gsm8k.py`: fixed-k GSM8K evaluation
- `Token2Token/eval_threshold_gsm8k.py`: batched catalyst/threshold evaluation
- `Token2Token/summarize_threshold_comparison.py`: quality/latency comparison
- `Token2Token/test_core.py`: 39 tests at last run
- `Token2Token/run_anchor_transition_v2.sh`
- `Token2Token/run_anchor_transition_v2_eval50.sh`
- `Token2Token/run_anchor_transition_v3_kl.sh`

## 13. Git History Landmarks

- `a78857d` anchor decode impact analysis/report
- `1ec2758` initial Token2Token folder
- `926cc8b` frozen-IG experiments
- `06cb459` rollout anchor selection
- `4b63612` local unlock selection
- `bdfa796` threshold-unlock training
- `d201094` after-before gain target
- `5186ecb` full GSM8K completion preservation
- `63a67cb` text-only catalyst filter
- `cb5b058` strict anchor-only first pass
- `5f0b61c` LTR residual cleanup
- `45201e5` chained base/trained evaluation
- `4c101f3` capped dynamic post-anchor transitions (V2)
- `14e57e4` KL-preserved smaller LoRA (V3)

## 14. Docker / tmux Usage

Run Docker commands on the host, not from inside the container:

```bash
cd /home/vishalg/Desktop/DhruveshProject
docker start confident_borg
docker exec -it -w /workspace/DhruveshProject confident_borg bash
```

To copy updated Token2Token code into the container, run on the host:

```bash
docker cp Token2Token confident_borg:/workspace/DhruveshProject/
```

Inside the container, tests are:

```bash
cd /workspace/DhruveshProject
python3 -m unittest Token2Token.test_core
```

Last test status before handover: 39/39 passed.

For a new background run, create tmux on the host, then invoke Docker there.
Do not run `docker ...` after entering the container because Docker CLI is not
installed inside it.

## 15. Final Research Takeaway

The empirical anchor phenomenon is real: a few carefully chosen gold anchors
can raise GSM8K accuracy by about 14 percentage points. The failure is turning
that cheating oracle into a learned decoder without damaging the base model.

The strongest implementation lesson so far is that anchor CE alone does not
train causal unlocking, while unconstrained post-anchor CE changes confidence
calibration and harms both quality and latency. KL preservation largely closes
the quality gap, suggesting the next useful direction is a gated or much more
strongly distilled post-anchor adapter rather than more epochs of the existing
global LoRA objective.
