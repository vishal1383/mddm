# Token2Token Research Log

Last updated: 2026-08-09

## Goal and acceptance rule

Train LLaDA-8B-Instruct to exploit out-of-order catalyst tokens and confidence
bursts. A useful result should improve both quality and throughput, or preserve
quality while improving throughput. A 1-2 percentage-point accuracy regression
is acceptable only for a meaningful speed improvement.

All comparisons must use the same examples, confidence threshold, numeric
threshold, and decoder policy. Small 16-example slices are only search gates.
Quality claims require at least 50-64 examples; on 64 examples, one lost answer
is a 1.56-point regression.

## Fixed-k proof of concept

- Base LLaDA, k=1: 995/1319 = 75.44%, 1 token/forward.
- Base LLaDA, k=2: 963/1319 = 73.01%, 2 tokens/forward.
- Earlier lookahead LoRA, k=2: 977/1319 = 74.07%, 2 tokens/forward.
- Interpretation: training recovered 14 answers relative to base k=2, but did
  not beat base k=1. This was evidence that out-of-order training can help, not
  a final result.

## Causal teacher-cache corrections

- The old cache's `unlocked` set included tokens already above tau before a
  catalyst. Training on it confused correlation with causal unlocking.
- The cache now records `newly_unlocked = correct_after - correct_before` and
  `newly_wrong = wrong_after - wrong_before`.
- Student unlock CE uses only `newly_unlocked`; confident mistakes can receive
  gold correction CE.
- Confidence margin applies to unlocked targets, not the catalyst itself.
- Catalyst ranking applies only to the catalyst, not every unlocked target.
- A 64-example teacher grid found a useful initial region at tau=0.90,
  candidate probability ratio=0.30, and wrong-unlock penalty=1. Mean causal
  correct gain was 1.906, mean new wrong was 0.469, and mean safe gain was
  1.438 per empty-canvas selection.

## Reduced-loop trials on IDs 7064-7079

Initial decoder: tau=0.90, `catalyst_filter=text`, force catalyst always, no
separate unlock forward. Training used 128 cached records and 140 updates unless
noted.

| Trial | Base correct | Trained correct | Base tokens/forward | Trained tokens/forward | Observation |
| --- | ---: | ---: | ---: | ---: | --- |
| correction=.25, margin=.25 | 9/16 | 11/16 | 3.664 | 3.625 | Quality improved, speed regressed slightly. |
| correction=0, margin=1.0 | 9/16 | 9/16 | 3.664 | 3.459 | Strong threshold margin made decoding slower. |
| first adapter at tau=.88 | 10/16 | 10/16 | 3.737 | 3.828 | Same quality, +0.091 tokens/forward; near miss. |
| first adapter at tau=.85 | 9/16 | 9/16 | 3.984 | 4.063 | Same quality and crossed 4, but only +0.079. |
| unlock-focused 120-step adapter at tau=.85 | 9/16 | 9/16 | 3.984 | 3.923 | Downweighting catalyst CE did not help. |

Observations:

- Correction supervision helped answer accuracy but did not by itself create
  more threshold crossings.
- Increasing the hard probability-margin weight made throughput worse. Raising
  cached gold probability is not equivalent to producing useful decode bursts.
- Reweighting toward unlocked CE and away from catalyst CE also slowed decode.
- The first adapter showed small positive speed at lower tau, but the effect was
  too small to promote.

## Decoder alignment: explicit below-threshold catalysts

The original `text` decoder usually chose a token already above tau. That token
would have been committed by threshold decoding anyway, so it often was not an
additional catalyst. The aligned `text-below` rule deliberately commits the
highest-confidence eligible text token below tau, then commits all threshold
tokens from the same model forward.

On IDs 7064-7079 at tau=0.90:

| Model | Correct | Tokens/forward | Observation |
| --- | ---: | ---: | --- |
| Base, text-below | 10/16 | 4.983 | Decoder alone was much faster than the old text schedule. |
| First causal adapter | 11/16 | 4.774 | +1 answer, but 18 more forwards than base. |
| Unlock-focused adapter | 10/16 | 4.911 | Tied quality, six more forwards than base. |
| Strong-margin adapter | 10/16 | 4.830 | Tied quality, 13 more forwards than base. |

The decoder change exposed a real anchor speed path, but old adapters were not
trained against a teacher that exactly matched this inference constraint.

## Inference-aligned 64-example cache

New cache:
`outputs/token2token/causal_unlock_search/gsm8k_train64_t090_q050_wp1_aligned_r4.jsonl`

Teacher candidate requirements:

1. Alphabetic text token only.
2. Base model currently predicts the gold token at that position.
3. Prediction confidence is below tau=0.90, matching `text-below` inference.
4. Gold probability is at least 0.50 times the best eligible gold probability.
5. Rank candidates by correct causal gain minus one times new confident errors.

Cache statistics:

- 64 records, 71 total teacher rounds.
- 123 causal newly correct threshold crossings and 87 new confident mistakes
  before filtering.
- Filtering for at least one causal unlock and selection score >=1 retained 28
  states: 97 newly correct tokens and 10 new confident mistakes.

## Aligned adapters

Four-pass adapter (80 updates, correction=.25, KL=10):

- IDs 7064-7079 without numeric safety: base 10/16 at 4.983; trained 10/16 at
  5.032. It saved four forwards.
- IDs 7080-7095 without numeric safety: base 12/16 at 5.769; trained 11/16 at
  5.902. It saved eight forwards but lost one answer.
- The lost example reasoned to 195 correctly, then a stray final numeric commit
  changed the extracted answer to 199.

## Numeric burst safety

Catalysts already excluded numbers, but threshold bursts could still commit
numeric predictions at tau=0.90. The decoder now optionally uses a separate
numeric threshold. Current policy: text tau=0.90, numeric tau=0.99.

On IDs 7080-7095:

- Base without numeric safety: 12/16 at 5.769 tokens/forward.
- Base with numeric tau=.99: 14/16 at 5.535 tokens/forward.
- Four-pass adapter with numeric safety: 13/16 at 5.705 tokens/forward.
- Numeric safety fixed the 195 -> 199 final-answer error.
- The remaining adapter-only failure was broader: it changed the copied sum
  `2+8+5+9` into `2+8+5+9+9`, producing 11 instead of 8. This suggested tiny
  cache overfitting, not only a final-answer suffix problem.

Gentle adapter configuration:

- Same 64-example aligned cache.
- Two passes / 40 updates.
- Catalyst CE=1, unlock CE=1, correction CE=.5, margin=.25, selection=.25.
- KL preservation=50, learning rate=1e-5.

Gentle adapter on IDs 7080-7095 with numeric safety:

- Base: 14/16 at 5.535 tokens/forward, 370 forwards.
- Trained: 14/16 at 5.611 tokens/forward, 365 forwards.
- Exact quality tie and five forwards saved.

## 64-example gate

Configuration: IDs 7064-7127, tau=.90, numeric tau=.99,
`catalyst_filter=text-below`, force catalyst always.

- Base: 47/64 = 73.44%, 1601 forwards, 5.117 tokens/forward.
- Gentle trained: 48/64 = 75.00%, 1593 forwards, 5.142 tokens/forward.
- Trained gained one answer and saved eight forwards.
- Paired churn: two base-only correct and three trained-only correct examples.
- Throughput delta was only +0.026 tokens/forward (about 0.5%), below the
  strict +0.10 gate. This is directionally positive but still undecided.
- Decision: increase aligned teacher data from 64 to 128 examples while keeping
  training exposure near 40 updates. This tests data diversity without adding
  more epochs or encouraging the earlier overfitting failure.

## Aligned 128-example cache

Cache:
`outputs/token2token/causal_unlock_search/gsm8k_train128_t090_q050_wp1_aligned_r4.jsonl`

- Same teacher configuration as the aligned 64-example cache.
- 128 records and 143 total teacher rounds.
- 247 causal newly correct threshold crossings and 215 new confident mistakes
  before filtering.
- The training filter (at least one causal unlock, selection score >=1) retained
  59 states from 43 records.
- Retained states contain 191 newly correct targets and 34 new confident
  mistakes; mean safe selection score is 2.661.
- Planned training is one pass, about 43 updates, with the gentle loss weights.
  This changes data diversity while holding optimization exposure nearly fixed.

## Aligned 128-data result

Adapter configuration:

- One pass over the 128-record aligned cache; training ended at 43 updates.
- Catalyst CE=1, unlock CE=1, correction CE=.5, margin=.25, selection=.25.
- KL preservation=50, learning rate=1e-5.
- Evaluation: train IDs 7064-7127, text tau=.90, numeric tau=.99,
  `text-below`, force catalyst always.

Paired 64-example result:

- Base: 47/64 = 73.44%, 1601 forwards, 5.117 tokens/forward.
- Trained: 50/64 = 78.12%, 1574 forwards, 5.205 tokens/forward.
- Delta: +3 correct answers, 27 fewer forwards, +0.088 tokens/forward.
- Relative throughput gain is about 1.7%; forwards/example fall from 25.016 to
  24.594, also about a 1.7% latency reduction under equal batch conditions.
- Paired churn: one base-only correct and four trained-only correct examples.
- The strict search gate marked this false only because its throughput cutoff
  is +0.10 tokens/forward. It passed quality and absolute-speed checks and
  missed the relative cutoff by 0.012.
- Interpretation: this is the first 64-example result improving both answer
  accuracy and decode efficiency. Promote it to a disjoint 64-example GSM8K
  test-split validation before any full 1319-example run.

## Disjoint GSM8K test-split validation

Configuration was unchanged except for evaluation data: GSM8K test IDs 0-63,
text tau=.90, numeric tau=.99, `text-below`, force catalyst always.

- Base: 43/64 = 67.19%, 1726 forwards, 4.746 tokens/forward.
- Aligned 128-data adapter: 41/64 = 64.06%, 1731 forwards,
  4.733 tokens/forward.
- Delta: -2 correct answers and five additional forwards.
- Paired churn: three base-only correct and one trained-only correct example.
- This is a 3.125-point accuracy regression and a small speed regression, so it
  fails the stated acceptance rule.
- Interpretation: the positive train-tail result did not generalize to the
  actual GSM8K test split. Do not launch a full 1319-example evaluation and do
  not scale the same objective blindly. The next experiment must address
  selection/generalization, not merely add more epochs.

## Iteration policy

- Reject clear failures on 16 examples.
- If direction is positive but undecided, increase aligned training data from
  64 to 128 examples, then to 256 only if needed.
- Promote a candidate to a 64-example gate before any full run.
- Promote to full GSM8K only when paired 64-example accuracy is within one
  answer and throughput improvement is stable.
- Persist raw predictions, adapters, cache files, manifests, and gate reports
  after every trial.

## External review and next ablation

Claude Code independently reviewed the cache selector, trainer, decoder, and
gate after the aligned-128 disjoint-test failure. Its main finding was a
measurable teacher-state quality shift rather than a block-size mismatch:

- The aligned-64 retained set had 97 causal correct unlocks and 10 newly
  confident mistakes, or 0.103 mistakes per correct unlock.
- The aligned-128 retained set at `selection_score >= 1` had 191 correct
  unlocks and 34 mistakes, or 0.178 mistakes per correct unlock. This is about
  73% more contamination.
- The selector sorts retained states by raw correct-unlock count, while the
  correction loss trains on mistakes produced by the oracle teacher rollout.
  Those corrections need not match mistakes encountered by the inference
  decoder, so the extra noisy states are a plausible source of train-tail
  overfitting.

Local verification on the existing aligned-128 cache found:

- `selection_score >= 2`: 38 states, 152 correct unlocks, 16 mistakes, ratio
  0.105.
- `selection_score >= 3`: 24 states, 120 correct unlocks, 12 mistakes, ratio
  0.100.

The next bounded experiment reuses the aligned-128 cache with
`selection_score >= 2`, otherwise retaining the gentle objective and exact
tau=.90 / numeric-tau=.99 / `text-below` matched decoder. It must pass both the
train-tail 64 and disjoint-test 64 comparisons before any scale-up. If it
fails, the next ablation is correction CE=0 on the original score>=1 set; do
not run both before observing the first result.

### Score>=2 train-tail result

- The score filter retained 38 states and produced 29 optimizer updates in one
  pass; no extra epochs were used.
- Base on train IDs 7064-7127: 47/64, 1601 forwards, 5.117 tokens/forward.
- Trained: 48/64, 1588 forwards, 5.159 tokens/forward.
- Delta: +1 correct answer, 13 fewer forwards, +0.042 tokens/forward (about
  0.8% fewer forwards).
- This is directionally positive but smaller than the failed score>=1
  train-tail result. The adapter is proceeding only to the predeclared
  disjoint-test 64 check, not to a full evaluation.

### Score>=2 disjoint-test result

- Base on GSM8K test IDs 0-63: 43/64, 1726 forwards, 4.746
  tokens/forward.
- Trained: 43/64, 1728 forwards, 4.741 tokens/forward.
- Delta: exact answer-accuracy tie and two additional forwards.
- Paired churn: one base-only correct and one trained-only correct example.
- Tightening teacher-state quality removed the previous two-answer test
  regression, supporting the cache-contamination diagnosis, but it did not
  create a speed gain. This is a clean null result and is not eligible for a
  full evaluation.
- Proceed to the predeclared correction-loss ablation: restore
  `selection_score >= 1` and set correction CE weight to zero. Evaluate the
  disjoint test slice first; run the train-tail slice only if test is
  promising.

### Correction-CE=0 disjoint-test result

- Training restored the 59 score>=1 states and completed 43 updates. All
  settings matched the failed aligned-128 adapter except correction CE weight
  was exactly zero.
- Base on GSM8K test IDs 0-63: 43/64, 1726 forwards, 4.746
  tokens/forward.
- Trained: 41/64, 1721 forwards, 4.760 tokens/forward.
- Delta: -2 correct answers and five fewer forwards. Paired churn was four
  base-only correct versus two trained-only correct examples.
- This exceeds the allowed quality regression and yields only about 0.3%
  fewer forwards. It fails both the quality rule and the meaningful-speed
  rule. The train-tail evaluation is skipped because the predeclared
  disjoint-test gate failed.
- Conclusion from the two ablations: noisy correction targets explain part of
  the original regression, but they are not the sole issue. The score>=2
  adapter preserved test quality but was speed-neutral; score>=1 without
  correction still lost test accuracy. The broader oracle-state versus
  inference-state mismatch remains the leading hypothesis.

### Correction-CE=0 and confidence-margin=0 result

Claude Code independently audited the saved predictions and identified the
same oracle/inference mismatch. It also found that the shared test regressions
were text substitutions or insertions rather than numeric-suffix errors. Its
second ranked ablation removed the confidence-margin loss after correction CE
had failed.

- Same 59 score>=1 states, one pass / 43 updates, attention-only rank-8 LoRA.
- Catalyst CE=1, unlock CE=1, correction CE=0, confidence margin=0,
  selection=0.25, KL=50.
- Base on test IDs 0-63: 43/64, 1726 forwards, 4.746 tokens/forward.
- Trained: 41/64, 1750 forwards, 4.681 tokens/forward.
- Delta: -2 answers and 24 additional forwards. Paired churn was three
  base-only correct versus one trained-only correct example.
- Reject. The margin term was not causing the speed failure; removing it made
  both quality and efficiency worse.

The next controlled capacity test keeps this softened objective fixed and
changes only LoRA coverage from attention projections to attention plus MLP
projections (`ff_proj`, `up_proj`, `ff_out`) at rank 8. This tests whether the
unlock objective lacks representational capacity. Rank 16 is conditional on
rank 8 preserving test quality and showing a speed signal; it is not launched
solely because VRAM is available.

### Full-projection LoRA rank-8 result

- LoRA targets: `q_proj,k_proj,v_proj,attn_out,ff_proj,up_proj,ff_out`.
- Rank 8, alpha 16, dropout .05: 22.016M trainable parameters, 0.274% of
  LLaDA-8B.
- All data, losses, seed, and decoder settings matched the narrow no-margin
  adapter exactly.
- Base on test IDs 0-63: 43/64, 1726 forwards, 4.746 tokens/forward.
- Trained: 43/64, 1745 forwards, 4.695 tokens/forward.
- Delta: exact quality parity but 19 additional forwards. Paired churn was one
  base-only and one trained-only correct example.
- Reject for speed. Broader MLP coverage recovered the narrow adapter's two
  lost answers, but did not create causal-unlock efficiency.

A rank-16 / alpha-32 run was launched after the rank-8 midpoint showed quality
parity and a provisional one-forward improvement. Alpha/rank remains 2. The
completed rank-8 result later removed that speed signal; rank 16 is retained
as the final requested capacity check, not as evidence for scale-up.

### Full-projection LoRA rank-16 result

- Same seven projection targets as rank 8.
- Rank 16, alpha 32, dropout .05: 44.032M trainable parameters, 0.546% of
  LLaDA-8B. Alpha/rank remained 2.
- Data, 43-update exposure, losses, seed, and decoder were otherwise identical
  to rank 8 and narrow no-margin.
- Base on test IDs 0-63: 43/64, 1726 forwards, 4.746 tokens/forward.
- Trained: 43/64, 1687 forwards, 4.856 tokens/forward.
- Delta: exact quality parity, 39 fewer forwards, +0.110 tokens/forward, and
  0.609 fewer forwards/example. This is about a 2.3% forward-count reduction.
- Paired churn: one base-only and one trained-only correct example.
- The predeclared gate passes quality, absolute speed, and the +0.10 relative
  tokens/forward threshold. This is the first disjoint-test candidate to pass
  all gates.

Decision: preserve the adapter and replicate on the next disjoint GSM8K test
IDs 64-127 before a full run. Do not claim a general result from the first 64
alone. If the second slice preserves quality within one answer and retains a
positive forward reduction, aggregate the 128 examples and promote the
candidate to larger validation.

### Rank-16 replication failure

The next test slice was evaluated with the same frozen adapter and decoder.
Base IDs 64-127 completed first at 50/64, 1584 forwards, and 5.172
tokens/forward. The trained evaluation was stopped after the predeclared small
gate became a clear failure on IDs 64-95:

- Base: 26/32, 728 forwards, 5.626 tokens/forward.
- Trained: 22/32, 785 forwards, 5.218 tokens/forward.
- Delta: -4 answers and 57 additional forwards.
- Paired churn: four base-only correct and zero trained-only correct examples.

The replication fails both quality and efficiency by large margins. The first
64 win is slice-specific and is not promoted to full evaluation or full
training. The partial predictions and completed second-slice base predictions
are preserved. Capacity alone is therefore rejected as the main solution.

Next direction: generate a small decoder-aligned teacher cache. Catalyst
positions must be selected by the actual highest-confidence `text-below`
policy rather than oracle lookahead, and threshold crossings must use text
tau=.90 plus numeric tau=.99. Validate that mechanism on small disjoint slices
before reconsidering full training.

## Decoder-aligned 32-record cache

The new backward-compatible cache mode uses the actual inference position
rule: highest-confidence alphabetic prediction below text tau=.90, with
leftmost cleanup only when no such position exists. Candidate effects use
numeric tau=.99 for digit-containing predictions. Existing oracle mode remains
the default.

Cache: `gsm8k_train32_t090_num099_decoder_r4.jsonl`.

- 32 records, 128 rounds, and one cleanup round.
- The decoder-selected catalyst prediction is gold-correct in 47/128 rounds
  and wrong in 81/128 rounds. This directly quantifies the oracle-cache gap.
- Across all rounds: 210 causal correct unlocks and 40 new confident mistakes.
- Filtering to at least one unlock and `selection_score >= 1` retains 63
  states with 202 correct unlocks and 17 mistakes.
- Of those useful states, only 25 catalysts are initially correct and 38 are
  wrong. The training target therefore needs to correct the position the
  decoder actually selects, rather than assume an oracle-correct catalyst.
- A stricter score>=2 set retains 34 states with 164 correct unlocks and eight
  mistakes.

Decision: ask for an independent loss audit, then run one matched 32-example
gate. Launch an overnight full cache/train/eval chain only if this mechanism
passes the small gate; do not commit compute merely because an earlier,
different adapter won on one favorable slice.

### Correct-catalyst single-anchor gate

Claude Code identified that unlocks measured after replacing a wrong selected
catalyst with gold are counterfactual: inference commits the predicted wrong
token. The trainer was patched to retain `prediction_correct_before` and
optionally require a correct catalyst. The conservative gate used score>=2,
correct catalysts only: 13 states from nine records, 50 unlocks, four
mistakes, one pass / nine updates.

Configuration: attention-only rank-8 LoRA; catalyst CE=1, unlock CE=1,
correction=.5, margin=.25, catalyst selection=.25, KL=50; matched test IDs
96-127, text tau=.90, numeric tau=.99.

- Base: 24/32, 856 forwards, 4.785 tokens/forward.
- Trained: 23/32, 875 forwards, 4.681 tokens/forward.
- Delta: -1 answer and 19 additional forwards.
- Paired churn: one base-only correct and zero trained-only correct.
- The first 16 temporarily showed a 12-forward saving, which reversed on the
  second half. This is another reminder not to promote partial batches.
- Reject. No overnight/full chain is launched from this objective.

Next: run a zero-training base-model diagnostic with the deployable top-two
`text-below` catalyst policy on the same 32 IDs. Build pair supervision only
if top two has a useful paired quality/forward tradeoff before training.

## Numeric-threshold cache mismatch

A subsequent code audit found that the aligned teacher cache labels every
correct crossing using tau=.90, while the matched evaluation decoder uses
tau=.90 for text and tau=.99 for digit-containing predictions. Static analysis
of the 191 retained causal unlock targets in the score>=1 cache found:

- 31/191 targets (16.2%) decode to digit-containing tokens.
- 26/191 targets (13.6% of all retained unlocks) have confidence in
  `[.90, .99)` and therefore would not be committed by the actual decoder.
- Examples include isolated `0`, `1`, `2`, `4`, and `5` tokens at confidence
  .91-.97.

This is a concrete train/inference mismatch in addition to oracle catalyst
selection. Any regenerated decoder-aligned cache must apply the same text and
numeric thresholds used at evaluation; simply scaling the current cache would
spend supervision on tokens that cannot contribute a decode-time burst.

## Base top-two catalyst diagnostic

The deployable decoder was changed only at the forced-catalyst action: when a
threshold burst stalls, commit the two highest-confidence eligible
`text-below` predictions instead of one. All other settings were matched:
LLaDA-8B-Instruct, GSM8K test IDs 96-127, completion length 128, text tau=.90,
numeric tau=.99, and threshold commits on the first forward.

- Top one: 24/32, 856 forwards, 4.785 tokens/forward.
- Top two: 24/32, 637 forwards, 6.430 tokens/forward.
- Delta: exact aggregate quality parity, 219 fewer forwards (25.6%), and
  +1.645 tokens/forward.
- Paired churn: one top-one-only and one top-two-only correct example.

This passes the small held-out gate by a large efficiency margin. It is a
decoder result, not yet evidence that pair-supervised LoRA training helps.
Because the earlier rank-16 result reversed on its next slice, first replicate
top two on the already-saved top-one test IDs 64-127. Promote to all 1,319
examples only if the 64-example accuracy is within one answer and forwards
fall by at least 20%.

In parallel, define a small teacher that follows the actual top-two inference
policy. The two positions must be selected jointly from the same model
forward, exactly as at inference; selecting one, re-forwarding, and then
selecting the second is a train/test mismatch. Treat exhaustive gold-token
pairs only as an oracle upper-bound diagnostic and never as deployable
training targets.

### Top-two 64-example replication

The replication used the saved top-one baseline and reran top two on GSM8K
test IDs 64-127 with all decoder settings matched.

- Top one: 50/64, 1,584 forwards, 5.172 tokens/forward.
- Top two: 48/64, 1,148 forwards, 7.136 tokens/forward.
- Delta: -2 answers, 436 fewer forwards (27.5%), and +1.964
  tokens/forward.
- Paired churn: five top-one-only and three top-two-only correct examples.
- On the contained IDs 96-127 subset, aggregate quality was tied; the entire
  -2 delta occurred on IDs 64-95. Fixed top two is therefore slice-sensitive.

Decision: it misses the predeclared within-one-answer quality gate and is not
promoted to the full 1,319 examples. Explore an adaptive second-anchor gate on
the same frozen base model: always place the first catalyst, but place the
second only when its confidence is sufficiently high relative to the first.
Compare every candidate against top one at the same text and numeric tau.

### Adaptive second-anchor gate

The selected rule always commits the highest-confidence catalyst, and commits
the second jointly selected catalyst only when its confidence is at least .60
and at least .85 times the first catalyst's confidence. Both positions still
come from one model forward; no gold information or re-forward is used to
choose the pair.

- Calibration, test IDs 64-95: top one 26/32 and 728 forwards; adaptive
  26/32 and 682 forwards.
- Validation, test IDs 96-127: top one 24/32 and 856 forwards; adaptive
  24/32 and 791 forwards.
- Combined: exact 50/64 quality parity, 1,584 -> 1,473 forwards (7.0% fewer),
  and 5.172 -> 5.561 tokens/forward.
- There was zero final-answer churn on both 32-example halves: every example
  kept the same correctness under top one and adaptive decoding.
- The second catalyst was accepted on 409/1,284 eligible catalyst cycles
  (31.9%), or 6.39 extra jointly committed catalysts per example on average.
- A looser .50/.80 gate retained the two-answer regression. A stricter
  .70/.90 gate tied quality on calibration but saved fewer forwards than the
  selected .60/.85 rule.

Decision: use .60 absolute / .85 relative as the overnight operating point.
Train a full one-epoch threshold-lookahead LoRA on all 7,473 GSM8K records.
The frozen teacher follows two consecutive deployed top-one catalyst cycles;
the student sees the pre-action canvas once and learns the second action plus
joint target ranking, with KL preservation. Evaluate 2k/4k/6k/final on a new
64-example test slice, select by the paired gate, then evaluate the selected
checkpoint and frozen base on all 1,319 test examples.

Independent code audit during training confirmed that catalyst selection,
same-forward burst commits, numeric tau, and leftmost cleanup are shared with
the deployed decoder, and that all labels come from frozen-base predictions
without gold-answer leakage. The intentional remaining mismatch is that the
teacher's second target comes from a post-burst second forward, while adaptive
inference must select it jointly from the original forward; the ranking loss
does not directly optimize the .60/.85 absolute gate. Checkpoint evaluation
must determine whether the distilled token actually clears that gate.

Reporting safeguards from the audit:

- Run full evaluation only if at least one checkpoint passes the paired
  64-example gate.
- Because test IDs 128-191 select the checkpoint, report both all 1,319 test
  examples and a separate untouched result on IDs 192-1318.
- A `GATE_FAILED` marker is distinct from `COMPLETE`; never mark a rejected
  checkpoint sweep complete as though it were a promoted result.

### Full threshold-lookahead training completion

- Completed 7,473/7,473 updates in 10,164.6 seconds (2h49m), one pass over all
  GSM8K training records with four canvases per record.
- Attention LoRA rank 8 / alpha 16, 8.39M trainable parameters (0.1045%).
- First 1,000 vs last 1,000 updates: transition CE .749 -> .602, selection
  loss 1.237 -> 1.130, future-target top-1 81.3% -> 84.7%, target-selection
  fraction 12.5% -> 18.4%, and joint success 2.0% -> 3.9%.
- Preservation KL remained bounded: .025 -> .034 average. No non-finite loss
  or interrupted checkpoint save occurred.
- Checkpoints 2k, 4k, 6k, and final are queued for the matched test-ID
  128-191 gate before any full evaluation.

First gate result, checkpoint 2k on test IDs 128-191:

- Base adaptive: 43/64, 1,512 forwards, 5.418 tokens/forward.
- Checkpoint 2k: 48/64, 1,141 forwards, 7.180 tokens/forward.
- Delta: +5 answers, 371 fewer forwards (24.5%), and +1.762
  tokens/forward.
- Paired churn: three base-only and eight trained-only correct examples.
- This independently clears both quality and efficiency gates. Continue the
  planned checkpoint comparison; do not stop at the first favorable checkpoint.

Complete checkpoint gate on test IDs 128-191:

| Model | Correct | Forwards | Tokens/forward | Delta vs base |
| --- | ---: | ---: | ---: | --- |
| Base adaptive | 43/64 | 1,512 | 5.418 | control |
| Checkpoint 2k | 48/64 | 1,141 | 7.180 | +5 answers, -371 forwards |
| Checkpoint 4k | 46/64 | 1,101 | 7.441 | +3 answers, -411 forwards |
| Checkpoint 6k | 45/64 | 1,146 | 7.148 | +2 answers, -366 forwards |
| Final | 46/64 | 1,062 | 7.714 | +3 answers, -450 forwards |

All checkpoints pass the paired quality/speed gate. Checkpoint 2k is selected
because checkpoint selection prioritizes correct answers before throughput;
its paired churn is three base-only vs eight trained-only correct examples.
The final checkpoint is faster but gives back two of checkpoint 2k's quality
gain, confirming that more optimization is not monotonically better.

### Merged checkpoint gate and full-run accounting

The deployable LoRA adapters were merged into the base weights before a second
checkpoint gate. This removes PEFT's extra adapter operations and makes each
trained forward comparable to a base-model forward. On test IDs 128-191:

| Merged model | Correct | Forwards | Tokens/forward | Eval wall time |
| --- | ---: | ---: | ---: | ---: |
| Base adaptive | 43/64 | 1,512 | 5.418 | 340.8s |
| Checkpoint 2k | 46/64 | 1,131 | 7.243 | 280.6s |
| Checkpoint 4k | 45/64 | 1,116 | 7.341 | 292.3s |
| Checkpoint 6k | 46/64 | 1,099 | 7.454 | 315.8s |
| Final | 46/64 | 1,115 | 7.347 | 250.7s |

Checkpoint 6k was promoted because it tied the best merged accuracy and used
the fewest model forwards. This ordering is only a 64-example selection
heuristic: merge-induced boundary changes are larger than the 32-forward gap
between the tied 2k and 6k checkpoints, so no intrinsic checkpoint-ranking
claim is warranted.

The gate also exposed a batching distinction that must remain explicit. The
1,512 -> 1,099 count is a 27.3% reduction in summed per-example forwards, but
batch size 16 requires 178 -> 161 whole-batch iterations, a 9.6% reduction,
and measured decode time changed by 7.3%. Final reporting therefore includes
all three views rather than calling tokens/forward a wall-clock speedup.

The frozen-base full adaptive run completed with 908/1,319 correct (68.84%),
30,985 summed per-example forwards, 5.449 tokens/forward, and 6,276.6 seconds.
On the untouched test IDs 192-1318 it has 771/1,127 correct (68.41%), 26,370
forwards, and 5.470 tokens/forward. The merged checkpoint-6k full run uses the
same batch composition and decode path; the only source change made between
the arms added adapter merge/loading metadata and did not alter decoding.

### Full checkpoint-6k result

The merged checkpoint-6k evaluation completed on all 1,319 GSM8K test
examples under exactly the same adaptive decoder as the frozen-base control.

| Slice | Model | Correct | Accuracy | Forwards | Tokens/forward |
| --- | --- | ---: | ---: | ---: | ---: |
| All 1,319 | Base | 908 | 68.84% | 30,985 | 5.449 |
| All 1,319 | Trained | 926 | 70.20% | 23,299 | 7.246 |
| Untouched 192-1318 | Base | 771 | 68.41% | 26,370 | 5.470 |
| Untouched 192-1318 | Trained | 785 | 69.65% | 19,872 | 7.260 |

On all test examples, paired churn is 91 base-only versus 109 trained-only
correct, exact McNemar p=.2292. The +1.36 percentage-point paired accuracy
delta has a bootstrap 95% CI of [-.76, +3.49] points. On the untouched slice,
churn is 79 versus 93, p=.3216, and the +1.24-point delta has a 95% CI of
[-1.15, +3.55]. These results do not establish an accuracy gain, but both
intervals exclude the predeclared -2-point quality tolerance.

Measured batch-16 decoding changed from 3,402 to 2,733 batch iterations
(-19.7%) and from 6,276.6 to 5,079.6 seconds (-19.1%). Summed per-example
forwards fell 24.8%; this larger number remains the logical batch-1 metric and
is not reported as measured wall-clock speedup.

Mechanistically, tokens/cycle rose from 5.449 to 7.246. Threshold-burst tokens
rose from 4.179 to 5.875 per cycle and account for 94.4% of that increase.
Second-catalyst acceptance also rose from 31.9% to 45.6%. The defensible read
is therefore that training primarily sharpened or unlocked larger confidence
bursts, with some increase in paired catalyst placement; it is not solely a
second-anchor effect.

Against earlier standard block-32 controls, the result occupies a new
quality/latency point but does not dominate them: k=2 reaches 73.01% in 132.1
minutes, while trained adaptive reaches 70.20% in 84.7 minutes; k=3 reaches
65.88% in 80.0 minutes. The trained model does Pareto-improve the matched
adaptive base (68.84%, 104.6 minutes), which is the direct causal comparison
for this training objective.
