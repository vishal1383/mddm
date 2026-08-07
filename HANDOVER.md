# MDDM Anchor / Token2Token Project Handover

Last updated: 2026-08-07 (second pass: decoder frontier + V4)

## 1. Current State

- Repository: `https://github.com/vishal1383/mddm`
- Active branch: `agent/token2token-anchor-training`
- Current implementation commit: `6d2441f`
  (`Pin numeric positions to the base distribution`)
- Base model throughout the latest work: `GSAI-ML/LLaDA-8B-Instruct`
- Main dataset: GSM8K (`openai/gsm8k`, `main`)
- Persistent Docker container: `confident_borg`
- Container project directory: `/workspace/DhruveshProject`
- Host project directory: `/home/vishalg/Desktop/DhruveshProject`
- tmux is not installed in the container. Background work is launched with
  `docker exec -d ... nohup`, and queued with the two chain scripts below.
- Two sequential queues are running on the one GPU:
  - `Token2Token/chain_remaining_work.sh`: round-2 decoder sweep, then V4a
    evaluation, then the full 1,319-example benchmark. Log:
    `outputs/token2token/chain_remaining.log`.
  - `Token2Token/chain_followup_work.sh`: the commit-phase mechanism
    experiment, then V4b. Waits on the first queue's PID. Log:
    `outputs/token2token/chain_followup.log`.
- **Do not write a waiter as `while pgrep -f <pattern>; do sleep; done`.**
  `pgrep -f` matches full command lines, including the waiter's own, so the
  condition never goes false. Three such waiters were queued and silently
  never fired. Wait on explicit PIDs instead, as the chain scripts do.
- The attempted V3 early-100 run was stopped before step 1. It produced only
  `outputs/token2token/anchor_transition_v3/train100_kl5/config.json`; there is
  no useful checkpoint to resume.

The working tree contains unrelated generated `__pycache__` changes, report
outputs, and `PapersMisc/RecentMDDMpaper.pdf`. Do not revert or stage these by
accident. All intended Token2Token source changes through V3 are committed and
pushed.

## 1b. Bottom Line (read this first)

Everything below was measured on base LLaDA-8B-Instruct with **no training**.
50 GSM8K test examples, 128-token completions, threshold 0.95 unless stated.

**The anchor idea is right. The machinery built around it was wrong.**

The project's premise was that placing one well-chosen token makes many other
positions confidently decodable. That is true, and the effect is large. The
mistake was spending a second model forward to collect the unlocked tokens: they
are still above threshold on the next forward and get committed there for free.

| Decoder | Accuracy | Forwards/example | Tokens/forward |
|---|---:|---:|---:|
| Catalyst, burst capped at 2 (the V2/V3 baseline) | 34/50 = 68% | 103.7 | 1.234 |
| Catalyst, two forwards, uncapped | 34/50 = 68% | 62.0 | 2.065 |
| Global-confidence top-k, k=1 | 29/50 = 58% | 128.0 | 1.000 |
| Global-confidence top-k, k=3 | 30/50 = 60% | 43.0 | 2.977 |
| Semi-autoregressive block, k=1 (**how LLaDA is normally decoded**) | 37/50 = 74% | 128.0 | 1.000 |
| **Semi-autoregressive block, k=3** | **38/50 = 76%** | **44.0** | **2.909** |
| **Single forward, one content anchor** | **36/50 = 72%** | **41.4** | **3.089** |

**The defensible claim: LLaDA can be decoded about 3x cheaper than its standard
schedule with no measurable quality loss, and there are two different ways to
get there.**

- Single-forward versus standard block k=1: 33 correct under both, 4 only under
  block, 3 only under single-forward, McNemar **p = 1.0000**, forwards
  **-86.56/example** (95% CI [-90.10, -83.06]).
- Block k=3 versus standard block k=1: 76% versus 74% at 44.0 versus 128
  forwards. Same conclusion, simpler mechanism.

**What is *not* claimed, and an earlier draft of this section wrongly implied
it.** At a matched forward budget, single-forward does **not** beat block
decoding. Block k=3 scores 76% at 44.0 forwards against single-forward's 72% at
41.4; paired, 5 versus 3 discordant, McNemar p = 0.7266, forwards +2.56 with a
95% CI of [-0.94, +6.10]. **The two are indistinguishable on both axes.** Block
k=3 is nominally ahead on accuracy and is the simpler decoder, so on this
evidence it is the better practical default.

The anchor work is not thereby refuted; it is scoped. Anchor choice is what
makes the *threshold* decoder work at all (section 8b: 72% versus 58% between
informative and uninformative anchors), and that remains the largest single
effect measured here. But the threshold decoder is not, on this evidence,
better than a well-configured block decoder.

The open question that follows is the obvious combination, which neither
baseline tries: block decoding commits a fixed k inside an ordered block, and
the threshold decoder commits adaptively but lets the burst run anywhere.
Round 4 runs blocks of 32 and 64 *with* adaptive threshold commits.

Note also the gap between the two top-k schedules: global-confidence k=1 scores
58% where block-structured k=1 scores 74%, both at 128 forwards. The decoding
*schedule* matters far more than the token budget. Quoting only the
global-confidence baseline, as the first version of this handover section did,
overstated the result by a wide margin.

**A separate result worth reporting on its own: block k=3 dominates block k=1
outright**, 76% at 44.0 forwards/example against 74% at 128. Nominally more
accurate at 2.9x the speed. LLaDA's usual one-token-per-forward block schedule
is not optimal even within its own family on this task, and simply raising k
inside the block is the single cheapest improvement available to anyone
decoding this model. That finding needs no anchors, no threshold, and no
training.

The Pareto front over all fourteen configurations measured:

| Operating point | Accuracy | Forwards/example | Tokens/forward |
|---|---:|---:|---:|
| Block k=3 | 38/50 = 76% | 44.0 | 2.909 |
| Single forward, 1 anchor, threshold 0.95 | 36/50 = 72% | 41.4 | 3.089 |
| Single forward, 1 anchor, threshold 0.90 | 34/50 = 68% | 35.7 | 3.585 |
| Single forward, 2 anchors, threshold 0.95 | 32/50 = 64% | 27.7 | 4.624 |

Two weaker comparisons, for orientation:

- Against the baseline the training work was actually being scored against
  (capped catalyst): 2.5x fewer forwards and +4 pp accuracy.
- Against *global-confidence* top-k at a matched budget: +12 pp for the same
  cost. This comparison is the weak one, since block-structured top-k is the
  real baseline.

Four things drove this, in order of size:

1. **Anchor quality is the dominant effect.** Forcing a content word unlocks
   2.089 further positions per forward; forcing the globally most confident
   token (usually whitespace or punctuation) unlocks 0.850. 72% versus 58%.
   Section 8b.
2. **The post-anchor unlock forward is counterproductive.** Removing it gains
   accuracy *and* a third of the forwards. Section 8b.
3. **The burst cap was handicapping the baseline**, so every earlier
   quality/latency comparison was made against a hobbled reference. Section 8b.
4. **Adaptive beats fixed.** Committing a variable number of positions, only
   where the model clears the threshold, beats committing a fixed k.

On the training line, V2 (44%) and V3 (64%) both failed, and the diagnosis is
in sections 8c and 8d: they supervised positions where the model confidently
preferred its own valid phrasing to the gold rationale wording, which taught it
to abandon coherent generation. V3 was never actually shown to lose quality
(paired McNemar p = 0.6875); its real defect was latency. V4 replaces that
objective; V4a ran but was too conservative to move any commit decision
(section 11b).

**Caveats that matter.** 50 examples cannot resolve differences of a few
answers: most of the accuracy comparisons here have McNemar p above 0.1. Only
the forwards/example numbers are tight. Wall-clock is contaminated by GPU
sharing. Completion length is 128, so absolute accuracies are not comparable to
published LLaDA numbers. The full 1,319-example benchmark is what turns the
headline into a claim.

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

## 8b. The Cap Was Hobbling the Baseline

V2 and V3 were both measured with `--max-threshold-tokens 2`. That cap was
introduced to stop one bad anchor corrupting dozens of positions, and it did,
but it also capped the baseline. The same catalyst decoder without the cap is
far faster at identical accuracy on the same 50 examples:

| Decoder | Accuracy | Forwards/example | Tokens/forward | Seconds/example |
|---|---:|---:|---:|---:|
| Catalyst, burst capped at 2 | 34/50 = 68% | 103.70 | 1.234 | 14.87 |
| Catalyst, burst uncapped | 34/50 = 68% | 62.00 | 2.065 | 10.26 |

Identical accuracy, 40% fewer forwards, 31% less wall time. Every V2/V3
quality/latency claim was made against the capped row, so the trained models
were being compared to a baseline that was already handicapped. **Do not use
the capped decoder as the baseline again.** The full-test uncapped base number
is 951/1319 = 72.10% at 2.2318 tokens/forward.

### Where the forwards actually go

Breakdown of the full 1,319-example uncapped base run, per example:

- 57.35 forwards, 128 tokens, 2.2318 tokens/forward
- 31.48 cycles: 25.93 catalyst cycles (2 forwards each) + 5.55 cleanup cycles
  (1 forward each)
- Tokens placed: 25.93 catalyst + 5.55 cleanup + 96.52 threshold burst

Two consequences:

1. The threshold burst places 75% of all tokens. The catalyst mechanism itself
   places 20%. Anything that raises burst density is worth more than anything
   that improves catalyst choice.
2. Cleanup forwards commit exactly one token and skip the burst entirely,
   because `unlock_active` is gated on `has_anchor`. They are 9.7% of forwards
   for 4.3% of tokens. `--commit-threshold-on-first-forward` fixes this.

### The decisive result: the unlock forward hurts

A catalyst cycle spends 2 forwards to place 1 catalyst plus 3.36 burst tokens.
The burst tokens that were *already* above threshold before the catalyst was
placed did not need the second forward at all. The `single_forward` arm
removes that forward entirely: one forward per cycle, committing the catalyst
plus every above-threshold position at once.

Base LLaDA, same 50 examples, threshold 0.95:

| Decoder | Accuracy | Forwards/example | Tokens/forward |
|---|---:|---:|---:|
| Catalyst, capped at 2 (old V2/V3 baseline) | 34/50 = 68% | 103.70 | 1.234 |
| Catalyst, uncapped, 2 forwards | 34/50 = 68% | 62.00 | 2.065 |
| Catalyst + first-forward commits, 2 forwards | 34/50 = 68% | 57.50 | 2.227 |
| Ordinary fixed top-k, k=1 | 29/50 = 58% | 128.00 | 1.000 |
| Ordinary fixed top-k, k=2 | 29/50 = 58% | 64.00 | 2.000 |
| **Single forward** | **36/50 = 72%** | **41.40** | **3.089** |

Single-forward is the only Pareto-optimal row: nothing else matches it on
either axis, let alone both.

The fixed top-k rows are the control that section 11.4 had been asking for
since the beginning, and they settle a question the project never tested: the
speedup is not simply "commit more tokens per forward". Ordinary k=2 commits
two tokens every forward unconditionally and lands at 58% for 64
forwards/example. Single-forward commits a variable number, only where the
model clears 0.95, and reaches 72% for 41.4. Paired, single-forward gains 11
and loses 4 (McNemar p = 0.1185) at -22.56 forwards/example, 95% CI
[-26.10, -19.06]. Committing *adaptively* is what buys both axes; committing
*more* does not.

#### The latency-matched control

Fixed top-k at k=3 spends 43.0 forwards/example, against single-forward's 41.4.
That is the same budget, so it isolates the decision rule from the speed:

| Decoder | Accuracy | Forwards/example | Tokens/forward |
|---|---:|---:|---:|
| Fixed top-k, k=3 | 30/50 = 60% | 43.0 | 2.977 |
| Single forward, one content anchor | 36/50 = 72% | 41.4 | 3.089 |

Paired: single-forward gains 10, loses 4, **-1.56 forwards/example** with a 95%
CI of [-5.10, +1.94], i.e. the budgets really are matched. McNemar
p = 0.1796, so on 50 examples this is suggestive rather than conclusive; the
effect is 12 points and the full benchmark is what settles it.

This is the cleanest form of the decoder claim. Given the same number of model
forwards, choosing *which* positions to commit -- one content anchor plus
everything already above threshold -- beats committing a fixed three per
forward by 12 points.

Note that k=1 also lands at 58%, using 128 forwards. Committing one token at a
time, the most conservative possible schedule, is not more accurate here --
it is three times slower for the same accuracy. **Do not read that as beating
LLaDA's own decoding.** Global-confidence top-k is not how LLaDA is generated;
the standard schedule is semi-autoregressive, filling the completion block by
block with confidence ordering only inside the active block, and it is a
stronger baseline. Round 2 measures it (`topk_block32_k1` as the quality
reference, `topk_block32_k3` latency-matched to single-forward). Until those
land, the honest claim is bounded: single-forward dominates the catalyst
decoder family and global-confidence top-k.

Paired against the two-forward decoder over the same 50 questions:

- Only two-forward correct: **0**
- Only single-forward correct: **2**
- McNemar two-sided exact p = 0.50 (not significant, but zero regressions)
- Forwards: **-20.56 per example**, bootstrap 95% CI [-23.74, -17.78]

Single-forward answered correctly every question the two-forward decoder did,
and two more. Against the original capped baseline this is 2.5x fewer forwards
and +4 pp accuracy, from a decode-schedule change with no training at all.

**Interpretation, and it inverts the project's core assumption.** The unlock
effect is real -- placing an anchor genuinely does push new positions above
threshold -- but those newly confident positions are confident *because of* the
anchor just placed. When the anchor is wrong, the burst inherits and amplifies
its error, and the decoder commits all of it irreversibly. Committing only
positions that were already confident *before* the anchor uses evidence that
does not depend on the anchor being right. That is why removing a forward
improves accuracy rather than trading it away.

This also explains the k-sweep in section 2, where greedy-IG accuracy peaks
around k=2 and then declines: the same error-amplification, driven there by
gold anchors placed too deep into the completion.

### What changes in the two examples that flip, and what that does not prove

Be careful with this section. **30 of the 50 questions produce different
completions**; the two decoders routinely take different paths through the same
reasoning. Only **two** change correctness, both in single-forward's favour.
In both of those, the two-forward decoder ends up with a wrong arithmetic
result:

- Example 9, gold 460. Two-forward writes `$10 x1.2 = $16 per hour`, then
  `$16 x 5 = $80`, and answers 480. Single-forward writes `$10 x 1.2 = $12`,
  then `$12 x 5 = $60`, and answers 460.
- Example 39, gold 18. Two-forward writes `3*2 = 12 miles per hour` and answers
  36. Single-forward writes `3 * 2 = 6 miles per hour` and answers 18.

The tempting story is that these are *dependent* tokens: a product only becomes
confident once its operands are on the canvas, which is exactly the class the
unlock forward exists to harvest, so the unlock forward commits it prematurely
and the error propagates.

**That story is consistent with the evidence but is not established by it.**
`Token2Token/divergence_analysis.py` finds that in both examples the
completions first diverge *earlier* than the arithmetic, on ordinary prose or
even whitespace (`run at 3*2` versus `run at 3 * 2`). So the wrong product may
be a downstream consequence of an earlier divergence rather than a
mis-timed unlock commit, and two examples cannot separate the two.

To actually test it, the decoder needs to record which phase committed each
position, so a wrong token can be attributed to a catalyst commit, a
first-forward threshold commit, or an unlock commit. That instrumentation does
not exist yet and is the concrete next step for this claim. The full
1,319-example run will also supply far more than two flipped examples to
categorise.

What the evidence does support without qualification is the aggregate result:
one forward per cycle is both cheaper and no worse, so **do not spend a second
forward harvesting confidence that the token you just committed created**.
Whether the mechanism is error amplification specifically remains open.

Report: `outputs/token2token/decoder_sweep/base50/paired_single_vs_two.md`.

### The anchor idea is confirmed; only its implementation was wrong

This is the most important result in this handover, and it separates the
project's *idea* from the *mechanism* built for it.

Both single-forward variants force exactly one token per forward. They differ
only in which token they are allowed to force: the most confident **alphabetic**
token, or the most confident token of any kind.

| Catalyst rule | Forced/forward | Threshold-unlocked/forward | Total | Accuracy | Forwards/example |
|---|---:|---:|---:|---:|---:|
| Alphabetic (`--catalyst-filter text`) | 1.000 | **2.089** | 3.089 | 36/50 = 72% | 41.4 |
| Any token (`--catalyst-filter any`) | 1.000 | **0.850** | 1.850 | 29/50 = 58% | 69.2 |

Paired over the same questions: text gains 11 and loses 4 (McNemar p = 0.1185)
at **-27.76 forwards/example**, 95% CI [-32.30, -23.66].

The forced commit count is identical, so the entire difference is in how many
*other* positions cross the confidence threshold as a result. Forcing a content
word unlocks 2.09 positions per forward. Forcing the globally most confident
token -- typically whitespace or punctuation the model is 99.99% sure of, and
which tells it nothing it did not already know -- unlocks 0.85.

**That is the anchor hypothesis, and it is confirmed.** Choosing which token to
commit first has a large causal effect on how many other positions become
confidently decodable. It is worth 2.5x in burst size and 14 points of
accuracy. The alphabetic filter, which looked like an incidental detail
inherited from the target-cache design, is doing most of the work.

What was refuted earlier is only the *implementation*: spending a dedicated
second forward to harvest the unlock. That is unnecessary, because the unlocked
positions are still above threshold on the next cycle's forward and get
committed there for free, and it is harmful, because the second forward commits
them one cycle earlier than the model would otherwise have to.

So the two findings compose rather than conflict:

- **Anchor selection: real and large.** Keep it. Improving it is the most
  promising direction left.
- **Anchor unlock harvesting: counterproductive.** Drop the second forward.

This also revises section 8b's reading. The threshold burst places 75% of the
tokens, but it only does so *because* a well-chosen anchor is placed each
cycle. The burst is the anchor's effect, not an independent mechanism.

#### Full anchor ablation

Four decoders, identical apart from how they choose the token they force each
forward. Base LLaDA, same 50 examples, threshold 0.95, one forward per cycle.

| Configuration | Accuracy | Forwards/example | Anchors/forward | Unlocked/forward | Tokens/forward |
|---|---:|---:|---:|---:|---:|
| Uninformative anchor (`--catalyst-filter any`) | 29/50 = 58% | 69.2 | 1.000 | 0.850 | 1.850 |
| Anchor only when the burst is empty (`--force-catalyst when-empty`) | 34/50 = 68% | 52.0 | 0.492 | 1.971 | 2.462 |
| **One content anchor** | **36/50 = 72%** | **41.4** | **1.000** | **2.089** | **3.089** |
| Two content anchors | 32/50 = 64% | 27.7 | 1.823 | 2.801 | 4.624 |

Three separate things are being varied and each matters:

1. **Anchor quality.** Informative versus uninformative, holding count at 1.000:
   unlocked/forward 2.089 versus 0.850, accuracy 72% versus 58%. This is the
   largest single effect in the whole investigation.
2. **Anchor presence.** Placing an anchor every forward versus only when the
   burst is empty: 3.089 versus 2.462 tokens/forward, 72% versus 68%. Note the
   burst barely changes (2.089 versus 1.971), so most of the throughput loss is
   simply the missing forced token, while the 4-point accuracy loss says the
   anchor commits are *good* commits, not merely extra ones.
3. **Anchor count.** More anchors buy throughput and cost accuracy, sub-linearly.

One nuance worth not overstating: `noforce` still reaches 1.971 unlocked per
forward while placing anchors only half as often. So the burst is not driven
purely by the anchor placed in that same forward; accumulated context carries
much of it. The anchor's marginal contribution is real but smaller than the
gap between informative and uninformative anchors implies.

#### Anchors have diminishing but real marginal returns

Forcing two content words per forward instead of one
(`--catalyst-tokens-per-forward 2`):

| Anchors/forward | Forced/fwd | Unlocked/fwd | Total | Accuracy | Forwards/example |
|---|---:|---:|---:|---:|---:|
| none informative (`any`) | 1.00 | 0.850 | 1.850 | 29/50 = 58% | 69.2 |
| one | 1.00 | 2.089 | 3.089 | 36/50 = 72% | 41.4 |
| two | 1.82 | 2.801 | 4.624 | 32/50 = 64% | 27.7 |

The first content anchor is worth about +1.24 unlocked positions over an
uninformative forced token. The second is worth about +0.87 more. Sub-linear,
because the second-most-confident content word is a weaker anchor than the
first, but far from exhausted. Forced/forward is 1.82 rather than 2.00 because
late in a decode there are often fewer than two eligible alphabetic positions
left.

The accuracy cost is real: 72% to 64%. So anchor count is a speed lever, not a
free win.

**It is, however, a better speed lever than lowering the threshold.** Two
anchors at threshold 0.95 give 64% at 4.624 tokens/forward; one anchor at
threshold 0.80 gives 62% at 4.558. Almost the same throughput, and the
multi-anchor route is no worse on accuracy. When more speed is needed, add
anchors before lowering the threshold.

Current Pareto front on the 50 examples:

| Operating point | Accuracy | Tokens/forward |
|---|---:|---:|
| one anchor, threshold 0.95 | 72% | 3.089 |
| one anchor, threshold 0.90 | 68% | 3.585 |
| two anchors, threshold 0.95 | 64% | 4.624 |

### Working hypothesis: the forced commit is the weak link

**This hypothesis was wrong, and the table above is why.** It is kept because
the reasoning that produced it is a trap worth seeing, and because one of its
predictions did hold for a reason unrelated to the mechanism proposed.

Every cycle commits one *forced* token -- the most confident eligible position,
whatever its confidence -- purely to guarantee the decode terminates. The other
commits are *threshold* commits, taken only above 0.95. These are not equally
reliable: the forced token is by construction the most confident position the
model judged **not** confident enough to commit.

Early round-2 data fits this. At threshold 0.99 the decoder scores 62.5% at
2.47 tokens/forward on the first 16 examples, worse than 0.95 on *both* axes.
Raising the threshold does not simply trade speed for accuracy; it suppresses
threshold commits, so a larger share of the completion comes from forced
commits, and accuracy falls with throughput.

If that is right, two round-2 arms should move in specific directions, and they
were queued before this was written:

- `single_forward_noforce` skips the forced commit whenever the threshold
  already selected something. It should **gain** accuracy, at some cost in
  forwards.
- `single_forward_any` drops the alphabetic restriction, so the forced token
  becomes the globally most confident masked position instead of the most
  confident *alphabetic* one. The alphabetic filter is what makes the forced
  commit risky: the best alphabetic candidate can sit far below the global
  best, while the global argmax is usually already above the threshold and
  would have been committed anyway. So `any` should behave much like
  `noforce` -- **more accurate, more forwards** -- and if it does, the
  text-anchor rule is not merely inert, it is actively costing accuracy.

  **Result: falsified, badly.** `any` scored 58% at 69.2 forwards/example
  against text's 72% at 41.4: worse on both axes, not better on one. The
  reasoning inverted the causation. It treated the forced commit as a cost
  paid for termination, when the forced commit is the anchor and is the
  reason the threshold burst exists at all. Because `any` almost always forces
  a token that was already above threshold, it forces something that would
  have been committed anyway, and so it never places the content word that
  makes other positions decodable. Its burst collapses from 2.089 to 0.850.

  The same inversion predicts `noforce` will be **worse**, not better: skipping
  the forced commit skips the anchor. That prediction is recorded here before
  that arm finished.

  **Result: the revised prediction holds.** `noforce` scored 68% at 52.0
  forwards/example, worse than 72% at 41.4 on both axes. The original
  hypothesis said it would gain accuracy; it lost 4 points and 26% throughput.

The arc is worth keeping as written: a hypothesis that explained one anomaly
(0.99 losing on both axes), a prediction from it that failed badly (`any`), the
inversion that failure forced, and a second prediction from the corrected
account that held (`noforce`). The corrected account is that the forced commit
is the anchor and the mechanism the project was built on, not overhead paid for
termination.
- Threshold 0.90 should **not** simply lose accuracy relative to 0.95, because
  it shifts work from forced commits to threshold commits.

If both hold, the lever that matters is the share of the completion placed by
threshold rather than forced commits, and V4's promote objective is aimed
correctly: pushing positions above the threshold reduces reliance on forced
commits. If they do not hold, prefer the simpler reading that 0.95 just
happens to sit near the optimum, and treat the V4 story with more suspicion.

#### Outcome: half right, and the half that failed matters

| Threshold | Accuracy | Forwards/example | Tokens/forward |
|---:|---:|---:|---:|
| 0.99 | 34/50 = 68% | 50.4 | 2.541 |
| 0.95 | 36/50 = 72% | 41.4 | 3.089 |
| 0.90 | 34/50 = 68% | 35.7 | 3.585 |
| 0.80 | 31/50 = 62% | 28.1 | 4.558 |

**Confirmed:** raising the threshold to 0.99 loses on *both* axes. A plain
speed-versus-quality view does not predict that; the forced-commit account
does, since a higher threshold suppresses threshold commits and shifts the
completion onto the forced token.

**Falsified:** lowering to 0.90 does not keep accuracy. It buys 16% throughput
and gives back the same two answers that 0.99 did.

So accuracy is U-shaped in the threshold with a peak near 0.95, while
throughput rises monotonically as the threshold falls. 0.95 was inherited from
the original setup but does sit near the accuracy optimum.

**The honest caveat is bigger than the finding.** Across the whole sweep
accuracy moves only between 34 and 36 correct out of 50. That is two examples;
50 examples cannot resolve it. What the sweep establishes reliably is the
*throughput* curve, which moves 2.54 -> 3.09 -> 3.59. Do not report the
U-shape as established until the full benchmark, which now runs 0.95 and 0.90
over all 1,319 examples in one process for exactly this reason.

Below 0.95 the trade is ordinary and steady: 0.80 gives up ten points of
accuracy for 47% more throughput. Three of the four thresholds sit on the
Pareto front (0.95, 0.90, 0.80); only 0.99 is dominated outright, which is the
part that needed explaining.

Two process notes on this sweep, both worth carrying:

- **Do not read partial runs.** At 8 examples 0.90 showed 75% and looked like
  a free win; at 16 it showed 56%; it finished at 68%. At 16 examples 0.80
  showed 44% and looked like a collapse; it finished at 62%. Every intermediate
  reading here was misleading in one direction or the other.
- A round-3 script extending the curve to 0.70/0.60/0.50 was written when 0.90
  looked like a free win (`Token2Token/run_decoder_sweep50_round3.sh`). It is
  queued at the end of the work rather than dropped: the thresholds are cheap
  (fewer forwards per example) and they complete the frontier, but nothing
  down there is expected to be a usable operating point.

### Report forwards/example, not wall seconds

There is one GPU (NVIDIA GB10). Several sweep arms were run while a training
job shared it, so their wall-clock numbers are inflated by contention and are
not comparable across arms. `single_forward` records 12.13 seconds/example
against `catalyst_uncapped`'s 10.26 despite using 33% *fewer* forwards, purely
because it ran under contention and the other did not.

Forwards/example is contention-independent and is the metric to quote. Only
quote seconds/example from arms that ran alone, and say so. The old section 9
criterion included a seconds/example bound, which is unsafe for exactly this
reason.

## 8c. Why V3 Lost: Digits, Not Language

Paired inspection of the 50-example V3 run against base (2 gained, 4 lost)
shows the losses are not degraded reasoning. V3 reproduces base's prose almost
verbatim and then slips a digit:

- Example 26: base `3 * $22.50 = $67.5`, V3 `3 * $22.50 = $67`; 243 becomes 242
- Example 12: base reaches 13, V3 reaches 3
- Example 16: identical opening sentences, 230 becomes 460

The anchor filter is alphabetic-only, so numeric tokens are never catalysts and
were moved purely as collateral. Numeric tokens carry the answer and have no
redundancy to absorb an error, which is why a 0.014-nat average KL drift still
cost four answers. Any future trainer should pin numeric positions to the base
distribution rather than let a global LoRA move them.

Note also that V3's `base_kl_loss` averaged 0.0144 against a total loss near
2.98, so KL weight 5.0 contributed about 0.07. The "strong base preservation"
was mostly nominal.

## 8d. Why the V2/V3 Objective Was Wrong

V3's `unlock_loss` settled at 3.63. That is the cross-entropy of gold at the
model's own top-two most confident post-anchor positions, so gold had roughly
2.7% probability there. Those positions are overwhelmingly not model errors:
they are places where the model confidently prefers its own valid phrasing over
the GSM8K gold rationale wording. Applying cross-entropy there teaches the
model to abandon its own coherent completion, which is the damage mechanism
behind both V2 (44%) and V3 (64%).

The lesson generalises: "differs from the gold token" is not the same as
"wrong". Only supervise a confident non-gold position when gold is still a live
alternative under the base model, and never treat gold rationale wording as
ground truth for phrasing.

## 9. Current Success Criterion

Superseded by section 8b. The capped-decoder thresholds below are kept only so
older results stay readable; they are not the bar any more.

- Old capped bar: at least 34/50 correct, under 5,185 forwards, under 14.87
  seconds/example.
- **Current bar, uncapped catalyst decoder, same 50 examples: at least 34/50
  correct, under 3,100 total forwards (62.0/example), under 10.26
  seconds/example.**

Two cautions on how to read any candidate against that bar:

1. **50 examples cannot resolve small quality differences.** At 68% the
   standard error is 6.6 pp, so 64% versus 68% is well inside one standard
   error. V3 was never shown to be worse than base on quality; it was shown not
   to be better, at a small latency cost. Use 50 examples only to reject large
   regressions, then confirm on 500+.
2. **A single fixed threshold is not a fair comparison.** Fine-tuning moves
   confidence calibration, so a trained model at 0.95 can trade quality for
   speed (or the reverse) purely through calibration drift and look like a win.
   The honest question is whether the trained accuracy/tokens-per-forward
   frontier sits above the base frontier across thresholds. Compare curves, not
   points: `Token2Token/run_pareto_benchmark.sh` sweeps 0.99/0.95/0.90/0.80 for
   both models under one decoder.

The ordinary fixed-`k` confidence baseline is now measurable through the same
harness with `--decoder topk --tokens-per-step K`, so the long-standing gap in
section 11.4 can finally be closed.

## 10. Compute Constraint (resolved)

This section used to state that the catalyst algorithm costs two forwards per
cycle -- one to place a catalyst, one to collect the burst -- capping it at 1.5
tokens/forward under a burst cap of two, and listed four possible escapes:
unlock four or more tokens per catalyst, place several catalysts in the first
forward, remove the second forward, or use speculative verification.

**The third escape was the answer, and the first two were also measured.**

- Removing the dedicated second forward: 3.089 tokens/forward at *higher*
  accuracy. Section 8b.
- Placing several catalysts in the first forward: works, sub-linearly, and
  costs accuracy. Two anchors give 4.624 tokens/forward at 64%.
- Unlocking four or more tokens per catalyst: not reached. One content anchor
  unlocks 2.089; two unlock 2.801 between them.
- Speculative verification: never tried, and no longer obviously needed.

The constraint that replaced it is different in kind: throughput is now set by
**how many positions clear the confidence threshold at once**, and that is
governed by anchor quality. See section 11.

## 11. Recommended Next Experiments

The original list here targeted the V2/V3 anchor-transition trainer and a
two-forward decoder. Both are superseded. What follows replaces it.

Priority order:

1. **Improve anchor selection.** This is the highest-value direction by a wide
   margin, because anchor quality is the largest measured effect: informative
   versus uninformative anchors is worth 2.089 versus 0.850 unlocked positions
   per forward and 14 points of accuracy. The current selector is a one-line
   heuristic (`text.strip().isalpha()`, most confident such token). Anything
   better goes straight to the bottom line. Round 4 tests cheap heuristics
   (minimum token length; restricting to below-threshold candidates). The
   principled version is to **learn** an anchor selector, which is the
   project's original idea and now has a decoder that rewards it.
2. **Settle why content anchors work.** Semantics or novelty? A content word may
   help because it is informative, or merely because it is a token the
   threshold would not have committed anyway -- the global argmax almost always
   would be. `--catalyst-filter below` separates these. The answer changes how
   a learned selector should be trained.
3. **Finish the full benchmark**, including the semi-autoregressive block
   baseline at a matched forward budget. Absolute accuracies at completion
   length 128 are not comparable to published LLaDA numbers; the comparison
   between arms is what carries.
4. **Then reconsider training.** V4's promote objective is aimed at the right
   quantity (positions crossing the threshold) but V4a moved gold probability
   without moving any commit decision. V4b loosens the throttles. If the
   learned-anchor-selector route from item 1 looks better, prefer it: it
   attacks the larger effect.

Still valid from the original list, and not superseded:

- **Numeric span handling.** Never place one isolated digit as an anchor.
  Either ban numeric anchors, as the current text-only filter does, or treat a
  contiguous number as one grouped action. Section 8c shows numeric damage is
  how V3 actually lost.
- **Threshold sweeps rather than a fixed threshold**, since fine-tuning moves
  calibration. Now standard in `run_pareto_benchmark.sh` and the full
  benchmark.
- **LM1B after GSM8K**, with likelihood-style quality against forward count.

## 11b. V4: Training for Parallel Decodability

`Token2Token/train_parallel_unlock.py` replaces the anchor-transition
objective. The reasoning is section 8b: the threshold burst places 75% of all
tokens, so tokens/forward is set by how many positions clear the threshold at
once, not by catalyst choice. So supervise that directly.

For every masked position of a realistic canvas, read the frozen base model's
own prediction and act only where a commit decision would change:

- **promote** — base already ranks gold first but sits under the threshold.
  Raising it converts a non-commit into a correct commit. This is the only
  bucket that buys throughput, and it is close to free: the model and gold
  already agree, so nothing is being overridden.
- **repair** — base is over the threshold on a non-gold token, so the decoder
  would irreversibly commit a mistake. Gated by `--repair-max-gold-rank` so it
  fires only where gold is still a live alternative, per section 8d. Defaults
  to weight 0.
- **preserve** — everything else, pinned to base with KL. Numeric positions are
  forced into this bucket regardless (section 8c).

Two details that matter:

- The promote objective is a **hinge**, not cross-entropy: it stops
  contributing once the position would commit. Cross-entropy keeps pushing
  toward probability 1 long after the threshold is crossed, which buys no extra
  tokens per forward and is exactly how V2 inflated confidence.
- The bucket assignment comes from the **teacher** (frozen base) while the
  hinge is measured on the **student**. The teacher decides where to act; the
  student decides how much more is needed.

Canvases are replayed from the cached threshold-gain trajectory, so they match
the partially-filled states the decoder actually visits rather than
random-mask denoising states.

Runner: `Token2Token/run_parallel_unlock_v4.sh`, fully environment-driven.

### V4a result: the objective works, the settings are too conservative

1,000 steps over 1,000 examples, promote weight 1.0, repair 0, preserve KL 20,
learning rate 1e-5, LoRA rank 8, numeric positions protected. 42 minutes.

- Promote hinge fell 0.257 -> 0.213 and flattened.
- Preserve KL stayed at 0.0024, so base behaviour was held tightly.
- About 15 promote positions per step, stable.

A hinge of 0.213 with target 0.97 implies the unsatisfied promote positions sit
at gold probability about 0.785. They started near 0.735. **They need to reach
0.95 to change a single commit decision**, so most of that movement bought
nothing: the run shifted probability without crossing the boundary that
matters. This is exactly what `promoted_fraction` was added to expose, and it
was added after V4a started, so V4a's log does not contain it.

Evaluated at threshold 0.95 on the matched 50 examples:

| Model | Accuracy | Forwards/example | Tokens/forward |
|---|---:|---:|---:|
| Base | 36/50 = 72% | 41.4 | 3.089 |
| V4a checkpoint 500 | 35/50 = 70% | 41.0 | 3.120 |
| V4a final (1000) | 36/50 = 72% | 41.5 | 3.086 |

**A precise no-op.** Same answers, same forwards, same throughput.

That is a better outcome than it first appears, and it should not be read as
"the objective does not work". V2 collapsed to 44% and V3 to 64%; V4a preserved
base behaviour *exactly*. The safety machinery -- a hinge that stops at the
commit boundary instead of cross-entropy, KL on every untouched position,
numeric positions pinned to base -- demonstrably holds the model in place. What
failed was the aim, not the safety.

That matters for what comes next: because the objective is now known not to
damage the model, aggressiveness can be raised without risking the V2/V3
collapse. The binding constraints were preserve KL 20 and learning rate 1e-5,
and the threshold it aimed at was unreachable. V5 loosens all three.

**Check the threshold sweep before spending more on this.** If base LLaDA at
threshold 0.90 already sits where a trained model at 0.95 would, training is
buying nothing that a decoder parameter does not, and V4 should be dropped
rather than tuned. Round 2 measures exactly that.

### The design error in V4, and V5

V4a was read as "the objective does not move anything". That reading is wrong,
and the mistake is worth stating plainly because it nearly closed a live line
of work.

V4a moved its promote positions from gold probability about 0.735 to about
0.785 and was judged a failure because nothing crossed **0.95**. But 0.95 is a
free parameter. It was chosen at the start of the project and never varied
during training; its only justification is that *base* LLaDA happens to peak
there (section 8b). A trained model has no reason to peak in the same place.
**At a decode threshold of 0.85, the positions V4a moved are already
committable.** The objective did work; it was measured against a threshold that
was never part of the design.

So V4 held the wrong axis fixed. It swept the objective and froze the
threshold, when the two have to be co-designed.

**V5 trains and decodes at a matched threshold.**
`Token2Token/run_threshold_matched_v5.sh`, swept by
`Token2Token/chain_v5_sweep.sh`.

The reasoning:

- Lowering the decode threshold is free throughput. Base goes from 41.4
  forwards/example at 0.95 to 35.7 at 0.90 and 28.1 at 0.80.
- What it costs is accuracy: 72% to 68% to 62%. Those losses are **wrong
  commits in the 0.80-0.95 confidence band**.
- That band is precisely what the promote and repair buckets are shaped to
  fix. Promote pushes gold-agreeing positions over the (now lower) bar; repair
  fixes positions committed wrongly just above it.
- Repair is also much safer at a low threshold. At 0.95, a confident non-gold
  prediction is usually the model preferring its own valid phrasing, which is
  what made repair dangerous (section 8d). At 0.85-0.95 the model is genuinely
  uncertain, so gold is a live alternative and the supervision is honest.

**Win condition:** reach base's 0.95 accuracy (72%, 36/50) while decoding at
0.85 or below, where base itself only manages roughly 65% but spends about 32
forwards/example instead of 41.4. That is the project's stated goal -- same
quality, better throughput -- stated as a measurable target for the first time.

The sweep covers the axes V4 never touched: threshold (0.80/0.85/0.90),
aggressiveness (preserve KL 5 versus 1, learning rate 3e-5 versus 1e-4), LoRA
capacity (rank 8 versus 32), and an ablation isolating promote from repair.

**Method note.** Do not conclude from a single failed training configuration
that a training direction is dead. V4a failed at one point in a large space,
with two throttles (preserve KL 20, learning rate 1e-5) and a threshold that
was never varied. That is one cell, not a verdict on the space.

### Generation-length caveat

Everything here uses `--completion-length 128`. Published LLaDA-8B-Instruct
GSM8K numbers use a longer generation budget, so the absolute accuracies in
this handover are not comparable to the paper's. All arms share the budget, so
the comparisons between them are valid; the absolute level is not a claim.

## 11c. New Decoder Knobs

All in `Token2Token/eval_threshold_gsm8k.py`; every default reproduces the
previous behaviour exactly, so older scripts still replay.

- `--commit-threshold-on-first-forward` — the catalyst/cleanup forward also
  commits its own above-threshold positions. Mainly recovers the wasted cleanup
  forwards of section 8b.
- `--no-unlock-forward` — drop the second forward entirely, one forward per
  cycle. Requires the flag above. This is the test of whether the unlock effect
  pays for its forward.
- `--catalyst-tokens-per-forward N` — commit the top N eligible text tokens in
  the first forward instead of one (handover section 10, bullet 2).
- `--adapter-scope unlock` — run the catalyst forward under `disable_adapter`,
  so catalyst selection and calibration stay exactly base and the adapter can
  only affect the post-anchor burst. This is section 11.1, the gated adapter,
  and it bounds the failure mode that sank V2.
- `--decoder topk --tokens-per-step K` — ordinary fixed-k confidence decoding
  through the same harness and metrics.

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
- `Token2Token/train_parallel_unlock.py`: V4 promote/repair/preserve trainer
- `Token2Token/summarize_decoder_sweep.py`: accuracy/latency table with a
  Pareto column
- `Token2Token/test_core.py`: 50 tests at last run
- `Token2Token/run_anchor_transition_v2.sh`
- `Token2Token/run_anchor_transition_v2_eval50.sh`
- `Token2Token/run_anchor_transition_v3_kl.sh`
- `Token2Token/run_decoder_sweep50.sh`: base decoder frontier, no training
- `Token2Token/run_parallel_unlock_v4.sh`: one V4 config, train then evaluate
- `Token2Token/run_pareto_benchmark.sh`: threshold sweep for base and trained

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

The empirical anchor phenomenon is real, and it does not need a cheating
oracle. The original experiment showed that a few *gold* anchors raise GSM8K
accuracy by about 14 points. The decoder sweep shows that anchors the base
model picks for itself, with a one-line heuristic and no training, are worth
about the same: 72% versus 58% against an uninformative anchor, and 2.089
versus 0.850 unlocked positions per forward.

So the project spent months trying to *learn* what turned out to be available
for free, while the machinery built to exploit it -- a second model forward to
collect the unlocked tokens -- was actively making things worse. Removing that
forward is a 2.5x reduction in model forwards against the baseline every
training run was scored against, at higher accuracy, with no training at all.

Three lessons worth carrying beyond this project:

1. **Measure the baseline under every knob you added for the trained model.**
   The burst cap existed to stop a trained model corrupting the canvas. It also
   halved the baseline's throughput, and nobody re-measured. Every
   quality/latency comparison for two model generations was against a
   handicapped reference.
2. **"Differs from gold" is not "wrong".** V2 and V3 applied cross-entropy at
   positions where the model confidently preferred its own valid phrasing to
   the gold rationale's wording. Gold probability there was about 2.7%. That
   objective taught the model to abandon coherent generation, and it is the
   whole explanation for 44% and 64%.
3. **Ablate the parts you think are incidental.** The alphabetic anchor filter
   was inherited from the target-cache design and looked like a detail. It
   turned out to be the largest single effect measured here. The dedicated
   unlock forward looked like the core mechanism, and was negative.

The most promising direction left is the project's original one, now with a
decoder that rewards it: **learn a better anchor selector**. Anchor quality is
the dominant lever, the current selector is a one-line heuristic, and nothing
about the earlier failures argues against learning selection -- they argue
against the transition objective that was wrapped around it.
