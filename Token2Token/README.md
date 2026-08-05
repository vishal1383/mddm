# IG Anchor Training V1

Minimal standalone anchor-supervision experiment for
`GSAI-ML/LLaDA-8B-Instruct`. Despite the existing folder name, V1 deliberately
does not train Token2Token transitions.

At each round the trainer:

1. Greedily ranks gold completion tokens by information gain (IG): insert each
   candidate at its gold position and measure its entropy reduction over the
   still-masked completion.
2. Takes the maximum-IG gold token, conditions on it, and recomputes IG to choose
   the next token, producing an ordered teacher sequence of up to five anchors.
3. Teacher-forces the previously selected gold anchors at their gold positions.
4. Trains the next IG-ranked anchor with a soft Gaussian placement prior,
   gold-relative-order loss, and token CE grounding loss.
5. Adds that gold anchor to the teacher-forced canvas and repeats.

IG is used only to construct training supervision. Inference uses the trained
model directly and does not pay the candidate-scoring cost.

The IG selection order determines which anchor is taught first. The relative-order
loss separately preserves the anchors' original left-to-right order in the gold
completion. Candidate anchors are limited to the first 75% of the completion to
exclude the far-right numeric failure mode from V1.

## Minimal anchor-order experiment

The simplified experiment removes the Gaussian placement and relative-order
losses. First, freeze base LLaDA and precompute the ordered greedy-IG gold
anchors:

```bash
python3 -m Token2Token.precompute_anchor_targets \
  --examples 7473 \
  --anchors 5 \
  --ig-batch-size 64 \
  --resume \
  --output outputs/token2token/anchor_targets/gsm8k_train.jsonl
```

Each saved record includes the source text, prompt IDs, complete gold token IDs,
and every anchor's IG rank, token ID, text, gold position, and IG score. Training
therefore never needs to recompute or change the target order.

Then train one LoRA update per example. Previous gold anchors are placed on the
canvas in greedy-IG order, and anchor CE supervises the next anchor token at its
gold position. After all five anchors are visible, completion CE supervises
every remaining masked gold token. The total is the sum of the mean five-anchor
CE and the mean remaining-sequence CE, deliberately giving the selected anchor
tokens more weight per token. All six canvases share one batched forward pass:

```bash
bash Token2Token/run_anchor_order.sh
```

This experiment has no Gaussian prior, placement marginalization,
relative-order loss, policy network, or Token2Token correction.

## Test

```bash
python3 -m unittest Token2Token.test_core
```

## GSM8K smoke run

```bash
MAX_STEPS=5 DATASET=gsm8k bash Token2Token/run_train.sh
```

Set `IG_BATCH_SIZE` above one when memory permits to score IG candidates faster.

## LM1B smoke run

```bash
MAX_STEPS=5 DATASET=lm1b bash Token2Token/run_train.sh
```

LM1B streams from `FrankCCCCC/lm1b` by default. Override it with
`--lm1b-dataset` when using a local mirror or another compatible `text` dataset.

Adapters and `train.jsonl` are written under `outputs/token2token/`.

## Inference

Inference uses ordinary confidence-based LLaDA decoding. There is no IG search,
anchor policy network, correction transition, or special regeneration stage.

```bash
python3 -m Token2Token.decode \
  --adapter-path outputs/token2token/llada8b_gsm8k_full/adapter-final \
  --prompt "A GSM8K question goes here" \
  --output outputs/token2token/example_decode.json
```

## Standard fine-tuning baseline

Train the same LoRA modules for one full GSM8K pass using ordinary
masked-denoising CE, without IG, Gaussian placement, or order loss:

```bash
python3 -m Token2Token.train_standard \
  --max-steps 7473 \
  --updates-per-example 1 \
  --output-dir outputs/token2token/standard_lora_gsm8k_full
```

## GSM8K k-sweep evaluation

For every `k`, normal confidence decoding commits the top `k` masked positions
per forward pass. The full launcher evaluates `k=1..5` for the IG-anchor LoRA,
the standard LoRA control, and unmodified LLaDA:

```bash
bash Token2Token/run_gsm8k_k_sweep.sh
```

Predictions are resumable, and the launcher writes the final comparison to
`outputs/token2token/eval_k_sweep/final_results.md`. Standard LoRA is the normal
masked-denoising fine-tuning baseline; it is not a causal left-to-right model.
