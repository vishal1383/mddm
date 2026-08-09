# All-Unlocked Adaptive Training V1

This experiment replaces checkpoint4000's fixed two-action target with the
complete cached unlock set.

## Training State

For each cache round, the input is the 128-token canvas immediately before the
gold catalyst is placed.

The target set is:

~~~text
T = {selected gold catalyst} union U_after
~~~

U_after contains every still-masked position whose frozen-base top-1 token is
gold-correct with confidence at least 0.95 after revealing the catalyst.
Targets outside the first 128 completion positions are omitted.
Records with no catalyst inside those 128 positions are skipped and reported
in the training log. In this cache, 7,472 of 7,473 records are eligible, so a
complete one-pass run performs 7,472 optimizer updates.

Each training example samples up to three cache rounds. Sampling always keeps
the earliest available state and the round with the largest target set.

## Objective

One student forward on the pre-anchor canvas minimizes:

~~~text
loss = target_set_CE + target_set_ranking + 5 * non_target_KL
~~~

- Target CE covers the catalyst and all U_after tokens.
- CE is averaged inside each cache round before rounds are averaged, preventing
  a rare large unlock burst from dominating the update.
- Ranking pushes the weakest exact target-token score above the strongest
  competing masked position with margin 0.1.
- KL preserves frozen base LLaDA at every masked position outside T.

The model is a rank-8 LoRA over q_proj, k_proj, v_proj, and attn_out.

## Matching Inference

Inference is adaptive rather than fixed-k:

1. Start from a fully masked 128-token canvas.
2. Run the model once over the full canvas.
3. Commit the highest-confidence alphabetic prediction as the catalyst.
4. From that same forward, commit every other top-1 prediction with confidence
   at least 0.95.
5. Repeat until complete.

There is no second post-anchor model forward and no fixed number of commits.
Every cycle commits at least one token, while its additional commit count is
determined by the 0.95 threshold.

## Data Available

Within the first 128 completion positions, the existing cache contains:

| Statistic | Value |
|---|---:|
| Applicable rounds | 161,831 |
| Unlocked token targets | 312,560 |
| Rounds with at least one unlocked token | 53.03% |
| Mean unlocked tokens per round | 1.93 |
| Mean targets including catalyst | 2.93 |
| Maximum unlocked tokens in one round | 90 |

No cache regeneration is required.

## Run

~~~bash
bash Token2Token/experiments/scripts/run_all_unlocked_v1.sh
~~~

The default run makes one pass over all 7,473 GSM8K records, skips the one
out-of-canvas record, saves every 500 updates, and then evaluates base LLaDA
and the final adapter on all 1,319 GSM8K test examples under the identical global
adaptive-0.95 decoder. It writes aggregate and paired quality/latency
comparisons before exiting. Set `EVAL_ALL_CHECKPOINTS=1` to evaluate all saved
intermediate checkpoints afterward.

The full evaluation defaults to batch size 16; both models use the same batch
size and decoding configuration.

Outputs:

~~~text
outputs/token2token/all_unlocked_v1/t095_full/
~~~

To serialize this run behind an evaluation already running in the container:

~~~bash
bash Token2Token/experiments/scripts/run_all_unlocked_after_eval.sh <evaluation-runner-pid>
~~~
