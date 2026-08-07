# Token2Token Parallel-Decoding Results

Last updated: 2026-08-07

## Bottom line

On LLaDA-8B-Instruct and the first 500 GSM8K test examples, a conservative
adaptive decoder slightly beats the standard blockwise `k=1` decoder while
using 44.4% fewer model forwards:

| Decoder | Correct | Accuracy | Forwards/example | Delta vs `k=1` |
|---|---:|---:|---:|---:|
| Standard block32 `k=1` | 383/500 | 76.6% | 128.0 | reference |
| Adaptive block32, threshold 0.99 | 385/500 | 77.0% | 71.2 | +0.4 pp |

The first 50 examples were used for exploration. On untouched IDs 50-499,
adaptive decoding scores 349/450 versus 346/450 for `k=1`, at 70.9
versus 128.0 forwards/example. Paired outcomes are 345 both correct, 1 only
`k=1` correct, 4 only adaptive correct, and 100 neither correct. The exact
McNemar p-value is 0.3750. The accuracy difference is not significant, while
the forward reduction is large and deterministic.

This is a 450-example held-out confirmation, not a full 1,319-example GSM8K
test-set result.

## Winning decoder

The successful rule is continuous with ordinary `k=1` decoding:

1. Restrict decoding to the leftmost unfinished 32-token block.
2. Run LLaDA once and commit the highest-confidence masked position.
3. From the same logits, also commit every remaining position in that block
   whose maximum token probability is at least 0.99.
4. Repeat until the 128-token completion canvas is full.

There is no content filter, gold information, IG search, policy network, or
adapter at inference. As the threshold approaches 1, the method becomes exact
standard blockwise `k=1`. The 0.99 operating point averages 1.798 committed
tokens per counted model forward on these 500 examples.

Run it inside `confident_borg` from `/workspace/DhruveshProject`:

```bash
python3 -m Token2Token.eval_threshold_gsm8k \
  --model-label llada8b_adaptive_block32_t099 \
  --thresholds 0.99 \
  --completion-length 128 \
  --batch-size 8 \
  --limit 500 \
  --commit-threshold-on-first-forward \
  --no-unlock-forward \
  --block-length 32 \
  --catalyst-filter any \
  --output-dir outputs/token2token/final/adaptive_t099
```

The matched reference is:

```bash
python3 -m Token2Token.eval_threshold_gsm8k \
  --model-label llada8b_block32_k1 \
  --decoder topk \
  --tokens-per-step 1 \
  --block-length 32 \
  --completion-length 128 \
  --batch-size 8 \
  --limit 500 \
  --output-dir outputs/token2token/final/base_k1
```

## Controls at 200 examples

| Configuration | Correct | Accuracy | Forwards/example | Held-out correct |
|---|---:|---:|---:|---:|
| Standard block32 `k=1` | 154/200 | 77.0% | 128.0 | 117/150 |
| Adaptive any-token, threshold 0.99 | 155/200 | 77.5% | 71.3 | 119/150 |
| Adaptive content-anchor, threshold 0.95 | 151/200 | 75.5% | 42.4 | 114/150 |
| Fixed block32 `k=3` | 138/200 | 69.0% | 44.0 | 100/150 |
| Lookahead LoRA checkpoint 125, fixed `k=2` | 147/200 | 73.5% | 64.0 | 110/150 |

The earlier apparent `k=3` win on the first 50 examples was selection noise:
it scored 38/50 there but fell to 100/150 held out. This is why the held-out
split is reported separately.

The adaptive 0.99 decoder changes only 11/500 extracted answers relative to
standard `k=1`. Four baseline errors become correct and two baseline-correct
answers regress, producing the net gain of two answers.

## Training result

The inference-aligned LoRA experiment distils two sequential frozen-base
`k=1` actions into one student forward. It trains on 500 GSM8K training prompts,
uses cached canvases plus the blank inference canvas, and preserves unaffected
masked-position distributions with KL. Checkpoint 125 matched `k=1` on the
50-example tuning slice, but it did not generalize:

| Split | Standard `k=1` | Trained fixed `k=2` | Forwards |
|---|---:|---:|---:|
| First 50 | 37/50 | 37/50 | 128 vs 64 |
| Held-out 150 | 117/150 | 110/150 | 128 vs 64 |
| All 200 | 154/200 | 147/200 | 128 vs 64 |

Therefore the current LoRA is not a successful model result. The successful
result is the adaptive base-model decoder. A direct target-selection ranking
loss is implemented as an opt-in next experiment, because the scaled run's
teacher-position selection metric declined even while future-token top-1
accuracy stayed high; that variant has not yet been evaluated and is not part
of the headline claim.

Train the reproduced lookahead setup with:

```bash
LOOKAHEAD=2 RECORD_LIMIT=500 MAX_STEPS=500 SAVE_EVERY=125 \
NAME=k2_train500_step500 \
bash Token2Token/run_online_lookahead_v6.sh
```

Enable the new selection-ranking term with
`SELECTION_LOSS_WEIGHT=1.0`; the default is zero so old runs remain exactly
reproducible.

## Validation artifacts

Compact summaries and paired reports are checked in under
`Token2Token/results/adaptive500/`.

The raw predictions, summaries, and paired reports are in the persistent
container under:

```text
outputs/token2token/online_lookahead_v6/validation200/base_k1/
outputs/token2token/online_lookahead_v6/validation200/base_k3/
outputs/token2token/online_lookahead_v6/validation200/base_adaptive_block32/
outputs/token2token/online_lookahead_v6/validation200/base_adaptive_any_block32_t099/
outputs/token2token/online_lookahead_v6/validation200/k2_train500_ckpt125/
outputs/token2token/online_lookahead_v6/validation500/
```

Run all unit tests with:

```bash
python3 -m unittest Token2Token.test_core
```

## Caveats

- Accuracy uses exact extracted GSM8K answers from 128-token completions.
- The primary efficiency measure is counted model forwards. Wall-clock timing
  depends strongly on batch composition and concurrent GPU work.
- The 0.99 threshold was chosen after the first 50-example exploration. IDs
  50-499 are the relevant held-out evidence.
- The remaining GSM8K test examples 500-1318 were not run in this validation.
