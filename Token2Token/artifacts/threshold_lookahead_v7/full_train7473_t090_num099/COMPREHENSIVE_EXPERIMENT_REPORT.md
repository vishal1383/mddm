# Learning Anchor Lookahead for Faster Diffusion Decoding

## Full GSM8K report

**Model:** GSAI-ML/LLaDA-8B-Instruct
**Dataset:** GSM8K (7,473 train examples; 1,319 test examples)
**Completion canvas:** 128 tokens
**Primary fixed-k checkpoint:** blockwise lookahead LoRA, checkpoint 4,000
**Primary adaptive checkpoint:** threshold-lookahead LoRA, checkpoint 6,000
**Evaluation:** full GSM8K test set

## Abstract

This work is inspired by the anchor-placement observation: on a masked diffusion
canvas, a small number of informative completion tokens can reveal several other
tokens. We call such a token an **anchor**. An anchor need not be the next
left-to-right token. It may occur in the middle or near the end of the completion,
but once it is present, the model can become confident about previously ambiguous
positions elsewhere on the canvas.

The central training idea is to make this reveal happen one forward earlier.
Frozen base LLaDA first takes one standard confidence action, recomputes the
canvas, and then takes a second action. The student sees only the canvas before
either action and is trained to make that second action available immediately.
At inference, two actions that normally require two sequential `k=1` forwards can
therefore be committed from one `k=2` forward.

The full-test fixed-k result is the cleanest proof of concept. Standard blockwise
confidence decoding with base LLaDA at `k=1` obtains **75.44% accuracy**, commits
**1 token per forward**, and uses 128 forwards per example. The lookahead LoRA at
`k=2` obtains **74.07% accuracy**, commits **2 tokens per forward**, and uses 64
forwards per example. This is exactly **2x logical token throughput and half as
many logical forwards for a 1.37 percentage-point accuracy loss**. At the same
`k=2` operating point, training improves accuracy from **73.01% for base LLaDA to
74.07%**, recovering 1.06 points of the quality lost by parallel placement.

The 2x claim is model-level throughput, not a claim of 2x measured hardware
speed. The fixed-k lookahead evaluation used an unmerged LoRA adapter, so its
169.6-minute wall clock was only slightly lower than the 172.9-minute `k=1`
baseline. In the later merged adaptive experiment, the trained model also produced
a realized speed gain: **70.20% versus 68.84% accuracy**, **7.246 versus 5.449
tokens per forward**, and **84.7 versus 104.6 minutes** under the same decoder.

The result supports a narrow but useful claim: anchor-inspired lookahead training
can expose future decoding actions early enough to improve the speed-quality
frontier. It does not yet establish a universally optimal decoder or a
statistically significant accuracy improvement.

## 1. Anchor idea recap

### 1.1 Diffusion decoding as canvas completion

LLaDA starts with a masked completion canvas rather than a strictly left-to-right
prefix. For a prompt `q`, completion length `L`, and vocabulary `V`, let

```text
x = (x_1, ..., x_L), where x_i is either MASK or a committed token.
```

One model forward produces logits `z_i(v)` for every position `i` and token
`v`. Excluding `MASK` as an output token, define

```text
yhat_i(x) = argmax_v softmax(z_i(x))_v
c_i(x)    = max_v softmax(z_i(x))_v.
```

Standard blockwise confidence decoding with `k=1` commits the prediction at the
single highest-confidence masked position in the leftmost unfinished block. A
`k=2` decoder commits the two highest-confidence positions from the same forward.
The second decoder is twice as aggressive, but the two predictions are made
without recomputing after the first token is inserted.

### 1.2 What is an anchor?

An anchor is a plausible token whose placement changes the surrounding canvas so
that additional positions become easy to decode. For a confidence threshold
`tau`, define the set of correct high-confidence positions under an offline gold
analysis as

```text
C_tau(x) = {j : x_j = MASK,
                 yhat_j(x) = gold_j,
                 c_j(x) >= tau}.
```

For a candidate gold intervention `(i, gold_i)`, write `x + (i, gold_i)` for the
canvas after inserting that token. Its unlock gain is

```text
gain(i | x) = |C_tau(x + (i, gold_i))| - |C_tau(x)|.
```

This formalizes the motivating observation: one token can reveal several others.
The intended speedup is obtained when a model learns to select such revealing
tokens early and then commits the newly confident tokens in parallel.

### 1.3 Full-test anchor intervention

The original intervention used greedy information gain to choose gold anchors,
placed `k=0,...,10` anchors, reset to the same initial canvas for every `k`, and
then completed the answer with the normal decoder. Every row below uses all 1,319
GSM8K test examples.

| Anchors k | Greedy-IG accuracy | Change from k=0 | Left-to-right prefix control |
|---:|---:|---:|---:|
| 0 | 54.9% | +0.0 pp | 54.9% |
| 1 | 66.7% | +11.8 pp | 54.8% |
| 2 | **68.8%** | **+13.9 pp** | 56.5% |
| 3 | 68.0% | +13.1 pp | 55.9% |
| 4 | 68.4% | +13.5 pp | 57.5% |
| 5 | 65.7% | +10.8 pp | 59.6% |
| 6 | 64.5% | +9.6 pp | 57.4% |
| 7 | 62.9% | +8.0 pp | 59.2% |
| 8 | 61.3% | +6.4 pp | 60.5% |
| 9 | 59.4% | +4.5 pp | 63.0% |
| 10 | 59.1% | +4.2 pp | 62.6% |

The first one or two anchors help substantially more than a same-length
left-to-right gold prefix. This shows that the model can use non-prefix structure.
The decline after the first few anchors also shows that an informative gold
intervention is not automatically a safe inference action.

### 1.4 Stray-anchor failure

Later IG ranks increasingly contain isolated digits, operators, punctuation, and
low-probability fragments. Forcing those tokens into a sparse canvas can redirect
the entire completion. A representative full trajectory is:

```text
k=0   __________                                        -> 1440 (wrong)
k=1   __(45: <<)_______                                 -> 1440 (wrong)
k=2   __(45: <<)____(71: She)__                         -> 2640 (correct)
...
k=9   several useful anchors, no final suffix digit     -> 2640 (correct)
k=10  ... (110: 0)                                     -> 26400 (wrong)
```

The last anchor is a gold token `0` at position 110, but in the generated path it
lands immediately after the already correct answer `2640`. The extracted answer
therefore becomes `26400`. Similar failures include `104 -> 1044`, copied
operands, arithmetic drift, and forced text fragments. The repeated lesson is:

```text
high intervention value != high probability != safe inference action.
```

## 2. Gaussian anchor-placement attempt

### 2.1 Intended model

The first training attempt tried to turn greedy-IG gold anchors directly into a
learned anchor-placement objective. For each training completion:

1. Greedy IG produced an ordered list of up to five gold anchor tokens.
2. Earlier gold anchors were teacher-forced into the canvas.
3. A Gaussian proposal centered near the next anchor's gold position provided a
   placement distribution.
4. The model was trained to predict the anchor token and preserve the relative
   order of anchor positions.

The implemented objective was approximately

```text
L_gaussian = L_place + 0.25 * L_relative_order + L_gold_token_CE.
```

Rank-8 LoRA was applied to `q_proj`, `k_proj`, `v_proj`, and `attn_out`.

### 2.2 Why it failed

The failure was structural rather than a matter of training for too few epochs.

1. **Gold IG was off-policy.** IG evaluates what would happen if a known gold
   token were forcibly inserted. Inference has neither the gold token nor the
   guarantee that the model assigns it meaningful probability.
2. **The Gaussian leaked the location.** Centering the proposal near the gold
   position made placement easier without teaching the model which currently
   plausible token should be selected from a blank canvas.
3. **Relative-order loss supplied almost no gradient.** Logged order losses were
   generally zero or numerically close to zero because the gold-centered proposal
   and teacher-forced anchors already preserved order.
4. **Unsafe anchors were supervised as if they were actions.** `<<`, operators,
   punctuation, and isolated digits could have large IG even when they were very
   unlikely under the unmodified model.
5. **Training and inference visited different states.** The trainer saw clean
   gold-conditioned canvases. The decoder visits canvases containing its own
   predictions and occasional mistakes.
6. **Normal decoding degraded after one epoch.** This is evidence of objective
   mismatch. More epochs would strengthen the wrong behavior rather than repair
   it.

The Gaussian experiment nevertheless identified the correct research question:
learn which token should be exposed early. The repair was to derive targets from
the actual decoder transition instead of from gold intervention order.

## 3. Decoder-aligned anchor lookahead

### 3.1 Core idea

Let `B` be frozen base LLaDA, `S_theta` be the same model with trainable LoRA, and
`pi` be a decoding policy. Starting from canvas `x`, frozen base performs two
sequential actions:

```text
a1 = pi(B, x)
x1 = Transition(x, a1)
a2 = pi(B, x1).
```

The student receives only `x`. It is trained so that both `a1` and `a2` are
selectable from one student forward. In words:

```text
teacher: x --forward--> a1 --update--> x1 --forward--> a2
student: x --------------------------one forward------> {a1, a2}
```

The first action is the anchor or catalyst. The second action is the token that
the anchor reveals. This is token-to-token **lookahead distillation**: the target
for the future action is a frozen-teacher prediction, not a gold IG token and not
a separately learned policy-network action.

### 3.2 Offline anchor-state cache

The final labels are computed online from frozen base LLaDA, but training benefits
from canvases at different completion stages. A train-only cache supplies those
starting states.

For each GSM8K training completion, the cache begins from a blank canvas and
considers alphabetic gold-token positions. A candidate is retained only if its
gold probability is at least `rho=0.7` times the best candidate gold probability
on that canvas. Each retained token is temporarily inserted and scored by the
increase in correct positions above `tau_cache=0.95` over the full canvas.

### Algorithm 1: Build unlock-state cache

```text
Input: frozen model B, prompt q, gold completion y,
       MASK token M, tau_cache=0.95, probability ratio rho=0.7
Output: a sequence of partial canvases X_cache

x <- [M, ..., M]
X_cache <- {x}

while x contains a masked alphabetic gold token:
    Run B(q, x).
    For each masked position j, compute top prediction yhat_j and confidence c_j.

    A <- {i : x_i=M, y_i is alphabetic, and
              P_B(y_i | q,x,i) >= rho * max_r P_B(y_r | q,x,r)}

    if A is empty:
        break

    for every candidate i in A:
        x_i <- x with gold token y_i inserted at i
        Run B(q, x_i)

        U_before(i) <- correct, still-masked positions other than i whose
                       top prediction under x has confidence >= tau_cache
        U_after(i)  <- correct, still-masked positions other than i whose
                       top prediction under x_i has confidence >= tau_cache
        W_after(i)  <- newly high-confidence positions under x_i whose
                       top prediction is not the gold token
        gain(i)     <- |U_after(i)| - |U_before(i)|

    i_star <- lexicographic argmax_i of
              (gain(i), |U_after(i)|, -|W_after(i)|,
               log P_B(y_i | q,x,i), -i)

    Save x and x with y_i_star inserted as possible training states.
    x[i_star] <- y_i_star
    For every j in U_after(i_star): x[j] <- y_j
```

This cache uses gold to generate useful state diversity, but cached anchors are
not the final CE labels. At each optimizer step, the teacher actions are recomputed
from the model's own predictions. The fully blank canvas is always included, and
up to three unique cached partial canvases are sampled per training record.

### 3.3 Exact teacher rollout

The method has two decoder-specific instantiations.

**Fixed-k teacher.** In the leftmost unfinished 32-token block, choose the masked
position with maximum model confidence, commit its predicted token, recompute, and
choose the next maximum-confidence position. This creates two sequential `k=1`
actions `(p1,t1)` and `(p2,t2)`.

**Adaptive threshold teacher.** Over the full 128-token canvas, choose the
highest-confidence alphabetic prediction below text threshold `tau_text=0.90` as
the catalyst. From the same logits, also commit every nonnumeric prediction above
0.90 and every numeric prediction above `tau_num=0.99`. Recompute once and repeat
to obtain the second catalyst and second confidence burst. If no alphabetic
catalyst remains, choose the leftmost unfinished prediction for cleanup.

### Algorithm 2: Two-cycle frozen-teacher rollout

```text
Input: frozen teacher B, prompt q, starting canvas x, policy pi, horizon H=2
Output: teacher actions a1=(p1,t1), a2=(p2,t2), initial teacher logits z_B(x)

x_work <- x

for h in {1,2}:
    z_h <- B(q, x_work) with the LoRA adapter disabled
    if h == 1:
        z_B(x) <- z_h

    At each masked position i:
        t_i <- argmax_v softmax(z_h[i,:])_v, excluding MASK
        c_i <- max_v softmax(z_h[i,:])_v

    if pi is fixed-k:
        p_h <- argmax confidence in the leftmost unfinished 32-token block
        a_h <- (p_h, t_p_h)
        x_work[p_h] <- t_p_h

    if pi is adaptive threshold:
        E <- masked positions whose top token is alphabetic and c_i < 0.90
        if E is nonempty: p_h <- argmax_{i in E} c_i
        else:             p_h <- leftmost masked position
        a_h <- (p_h, t_p_h)

        B_h <- all other masked nonnumeric positions with c_i >= 0.90,
               plus all other masked numeric positions with c_i >= 0.99
        Commit a_h and every prediction in B_h using the same z_h

return a1, a2, z_B(x)
```

Both target tokens are teacher-generated. They can differ from the GSM8K gold
completion. No gold answer, IG calculation, or policy network is used in this
rollout.

### 3.4 Student losses

For student logits `z_theta(x)`, let

```text
l_theta(i,v | x) = log softmax(z_theta(x)[i,:])_v.
```

Let `E(x)` be the still-masked positions eligible under the relevant block or
full-canvas policy. The student receives the original canvas `x`, before `a1` was
placed.

**Future-token cross-entropy.** The future action `a2=(p2,t2)` is trained one
cycle early:

```text
L_future(x) = -l_theta(p2,t2 | x).
```

There is no separate CE term for `a1`: it is already the action frozen base can
take from `x`. There is also no direct gold-answer CE in this lookahead objective.

**Joint selection ranking.** Correct token identity is insufficient if `p2` does
not rank among the positions the decoder will commit. Define

```text
g_target = min(l_theta(p1,t1 | x), l_theta(p2,t2 | x))
g_comp   = max over i in E(x) excluding {p1,p2}
           of max_v l_theta(i,v | x)

L_rank(x) = softplus(g_comp - g_target + margin), margin=0.1.
```

This pushes the weaker of the two teacher targets above the strongest competing
position. It is what connects token prediction to the decoder's top-position
selection rule.

**Base-preservation KL.** To prevent the LoRA from changing the rest of the
language model unnecessarily,

```text
L_KL(x) = mean over masked i != p2 of
          KL(P_B(. | q,x,i) || P_theta(. | q,x,i)).
```

The first target position `p1` remains inside the KL term because the base already
predicts it from `x`. Only the future position `p2` is removed so its CE can move
freely.

**Total loss.** Averaged over the blank canvas and sampled cached states:

```text
L_total = 1.0 * mean(L_future)
        + 1.0 * mean(L_rank)
        + 5.0 * mean(L_KL).
```

### Algorithm 3: One optimizer step

```text
Input: one GSM8K training pair (q,y), cache states for that record

X <- {fully masked 128-token canvas}
     union up to 3 randomly sampled unique cached canvases

Disable LoRA adapter.
For every x in X:
    (a1, a2, z_B(x)) <- FrozenTeacherRollout(B, q, x, H=2)

Enable LoRA adapter.
Run S_theta(q,x) once for every original x in X.

For every state x:
    compute L_future from a2
    compute L_rank from {a1,a2} and the hardest competing position
    compute L_KL against z_B(x) away from the future target

L_total <- mean(L_future) + mean(L_rank) + 5*mean(L_KL)
Backpropagate through LoRA parameters only.
Clip gradient norm to 1.0 and update with AdamW.
```

### 3.5 Optimization configuration

The adaptive checkpoint-6,000 run used the following full-data configuration.

| Setting | Value |
|---|---|
| Base model | LLaDA-8B-Instruct |
| Training examples / steps | 7,473 / 7,473 (one epoch) |
| Canvas / block length | 128 / 128 |
| Teacher lookahead | 2 adaptive catalyst cycles |
| States per example | 1 blank + up to 3 cached states |
| Text / numeric threshold | 0.90 / 0.99 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| LoRA modules | `q_proj,k_proj,v_proj,attn_out` |
| Learning rate | 3e-5 |
| Precision | BF16 |
| Gradient clipping | 1.0 |
| Loss weights | future 1, ranking 1, KL 5 |
| Ranking margin | 0.1 |
| Seed | 42 |

The fixed-k checkpoint-4,000 used the same two-action loss family but rolled out
two sequential standard `k=1` actions in the leftmost unfinished 32-token block.
Its intended inference policy was therefore fixed `k=2`.

### 3.6 Fixed-k inference

### Algorithm 4: Blockwise fixed-k decoder

```text
x <- [MASK, ..., MASK] with length 128

while x contains MASK:
    Run the student once on (q,x).
    Find the leftmost unfinished 32-token block.
    For every masked position i in that block, compute top token and confidence.
    Select the k positions with highest confidence.
    Commit all k predicted tokens simultaneously from the same forward.

return decoded completion x
```

`k=1` is standard confidence decoding. `k=2` is the trained lookahead operating
point. There is no threshold, IG computation, gold token, or teacher call at
fixed-k inference.

### 3.7 Adaptive inference

### Algorithm 5: Two-catalyst adaptive decoder

```text
Parameters: tau_text=0.90, tau_num=0.99,
            max catalysts K=2,
            second-catalyst minimum confidence=0.60,
            second/first confidence ratio=0.85

x <- [MASK, ..., MASK] with length 128

while x contains MASK:
    Run the student once on (q,x).
    Compute each masked position's top token and confidence.

    E <- alphabetic top-token positions with confidence < tau_text
    if E is nonempty:
        a1 <- highest-confidence position in E
        optionally choose the next position a2 in E only if
            confidence(a2) >= 0.60 and
            confidence(a2) / confidence(a1) >= 0.85
        commit a1 and, when accepted, a2
    else:
        commit the leftmost unfinished prediction

    From the same logits, also commit every remaining nonnumeric prediction
    with confidence >= 0.90 and every remaining numeric prediction with
    confidence >= 0.99.

return decoded completion x
```

No extra forward occurs between catalyst placement and the confidence burst.
This is essential: the trained model must expose the additional work in the same
forward for the speedup to be real. Alphabetic catalyst filtering and the 0.99
numeric threshold explicitly guard against the stray-number failure seen in the
IG experiment.

## 4. Results

### 4.1 Primary fixed-k proof of concept

These are full 1,319-example GSM8K test results with a 128-token completion and
32-token block schedule.

| Model and decoder | Correct | Accuracy | Tokens/forward | Forwards/example | Eval wall clock |
|---|---:|---:|---:|---:|---:|
| Base LLaDA, standard confidence `k=1` | 995/1,319 | **75.44%** | 1.000 | 128 | 172.9 min |
| Base LLaDA, fixed `k=2` | 963/1,319 | 73.01% | 2.000 | 64 | 132.1 min |
| Lookahead LoRA checkpoint 4,000, fixed `k=2` | 977/1,319 | **74.07%** | **2.000** | **64** | 169.6 min |

The two comparisons answer different questions.

1. **Lookahead `k=2` versus standard `k=1`:** logical throughput doubles from
   1 to 2 tokens per forward and forwards halve from 128 to 64, while accuracy
   changes from 75.44% to 74.07%, a loss of only 1.37 points. The paired test has
   902 both-correct, 93 `k=1`-only correct, and 75 lookahead-only correct examples;
   exact McNemar `p=0.1895`.
2. **Lookahead `k=2` versus base `k=2`:** training raises accuracy from 73.01% to
   74.07%, a gain of 1.06 points at the same 2 tokens per forward.

The unmerged adapter made the lookahead row expensive in wall-clock terms. It
therefore demonstrates a 2x **logical** decoding rate, not a 2x end-to-end hardware
rate. Base `k=2` confirms that the evaluator itself benefits from fewer forwards;
merging the adapter is required for a clean fixed-k hardware benchmark.

### 4.2 Matched adaptive result

The later checkpoint-6,000 experiment compares frozen base and lookahead LoRA
under exactly the same adaptive decoder from Algorithm 5. The adapter was merged.

| Model | Correct | Accuracy | Tokens/forward | Forwards/example | Canvas tokens/s | Eval wall clock |
|---|---:|---:|---:|---:|---:|---:|
| Base LLaDA | 908/1,319 | 68.84% | 5.449 | 23.49 | 26.90 | 104.6 min |
| Threshold-lookahead LoRA | **926/1,319** | **70.20%** | **7.246** | **17.66** | **33.24** | **84.7 min** |

Relative to the matched base decoder, the trained model produces:

- +1.36 percentage points in observed accuracy;
- +33.0% tokens per forward;
- -24.8% summed per-example logical forwards;
- +23.6% measured canvas tokens per second;
- -19.1% evaluation wall clock.

The accuracy bootstrap 95% interval is `[-0.76,+3.49]` percentage points and
exact McNemar `p=0.2292`, so the accuracy change is positive but not statistically
conclusive. The efficiency improvement is the stronger claim.

### 4.3 Why the adaptive model is faster

| Commit mechanism | Base LLaDA | Threshold-lookahead LoRA |
|---|---:|---:|
| Total tokens per cycle | 5.449 | 7.246 |
| Threshold-burst tokens per cycle | 4.179 | 5.875 |
| Second-catalyst acceptance | 31.9% | 45.6% |

Threshold bursts explain 94.4% of the tokens-per-cycle increase. This matches the
anchor narrative: training does not merely force an extra token. It changes the
initial canvas distribution so that more other positions cross the safe commitment
threshold in the same forward.

### 4.4 Ordinary fine-tuning control

Ordinary denoising LoRA used the same base model, GSM8K split, one epoch, LoRA
rank, alpha, dropout, and target modules. It sampled a random mask ratio from 0.15
to 1.0 and minimized gold CE at every masked position. It had no anchor,
lookahead, ranking, or preservation objective. Under the completed text-catalyst
`tau=0.95` control:

| Model | Correct | Accuracy | Tokens/forward | Eval wall clock |
|---|---:|---:|---:|---:|
| Base LLaDA | 956/1,319 | 72.48% | 3.286 | 135.9 min |
| Ordinary denoising LoRA | 728/1,319 | 55.19% | 3.008 | 162.9 min |

The ordinary LoRA used learning rate `1e-4`, while lookahead LoRA used `3e-5`, so
this is not a strict objective-only ablation. It does show that generic GSM8K
fine-tuning does not automatically produce the observed throughput behavior.

## 5. Interpretation

### Supported by the completed experiments

- Out-of-order completion tokens can reveal useful information that a same-length
  left-to-right prefix does not reveal.
- Direct gold-IG supervision with Gaussian placement is off-policy and can damage
  normal decoding.
- Two-step decoder-aligned distillation can make a future action available one
  forward earlier.
- At fixed `k=2`, lookahead training recovers part of the quality lost when moving
  from `k=1` to parallel placement.
- Under the matched adaptive decoder, training increases tokens per forward and
  reduces measured wall clock without an observed accuracy loss.
- Filtering forced anchors to alphabetic text and requiring 0.99 confidence for
  numbers directly addresses the documented stray-number failure.

### Not yet established

- A statistically significant accuracy gain.
- A 2x end-to-end hardware speedup; the demonstrated 2x fixed-k result is logical
  token throughput.
- Dominance over every threshold and fixed-k operating point.
- Generalization beyond GSM8K.
- Robustness across multiple seeds.
- Correction of an anchor after it has already been committed.

## 6. Next steps

### 6.1 Increase k and lookahead horizon

The immediate extension is horizon `H=3`, followed by `H=4` only when the shorter
horizon improves a held-out speed-quality gate. A horizon-`H` teacher produces
sequential actions `a1,...,aH`; the student receives the initial canvas and ranks
all actions while applying CE to future actions `a2,...,aH`. Inference can then
attempt `k=3` or a dynamically gated number of anchors.

Blindly increasing fixed `k` is unsafe because teacher errors compound. Each extra
action should need an absolute confidence gate, a confidence ratio relative to the
first anchor, and token-type safety checks.

### 6.2 Treat anchor choice as short-horizon dynamic programming

Anchor selection can be framed as planning over canvas states without learning a
separate RL policy network. Let a state `s` be the current canvas and an action
`A` be a set of one to `K` plausible token placements. Define a short-horizon
reward

```text
R(s,A) = newly unlocked safe tokens
       - beta * newly confident mistakes
       - lambda * estimated model-forward cost.
```

The finite-horizon objective is

```text
V_h(s) = max_A [R(s,A) + V_{h-1}(Transition(s,A))].
```

Exact search over all token subsets is unnecessary. Keep the top few plausible
anchors, expand only one- and two-anchor actions, and use beam search or approximate
dynamic programming for two to four steps. Frozen model probabilities supply both
the transitions and rewards. The best short plans become distillation targets for
the same CE, ranking, and KL objective.

### 6.3 Add true Token2Token correction

The current model only performs `MASK -> token` transitions. Once a wrong anchor
is committed, it cannot be revised. The next model should support

```text
wrong token -> corrected token
wrong token -> MASK
wrong suffix token -> deletion/end-of-answer
```

Training data can be generated from the model's own rollouts:

1. Decode partial canvases and retain states containing committed mistakes,
   especially numeric suffixes and low-confidence anchors.
2. Give a frozen teacher additional context or a later rollout state.
3. Label positions where the teacher changes its preferred token.
4. Train replacement CE at those positions, rank correction actions alongside new
   anchor actions, and add a penalty for unnecessary edits.
5. At inference, reserve a small revision budget and allow the decoder to replace
   or remask a committed token when the correction score exceeds the next-anchor
   score.

This extension directly targets failures such as `2640 -> 26400`. A deletion or
remasking action can remove the stray suffix instead of forcing the rest of the
completion to accommodate it.

### 6.4 Complete the experimental controls

- Merge the fixed-k adapter and repeat `k=1`, `k=2`, and `k=3` to measure actual
  hardware throughput without adapter overhead.
- Evaluate base, ordinary LoRA, and lookahead LoRA under identical thresholds and
  decoder rules.
- Repeat ordinary LoRA at learning rate `3e-5` for a strict objective ablation.
- Add active-row batch compaction so completed examples stop consuming padded
  batch iterations.
- Test LM1B or another non-math corpus and run multiple seeds.

## Artifact locations

- Current adaptive report: `Token2Token/artifacts/threshold_lookahead_v7/full_train7473_t090_num099/FINAL_REPORT.md`
- Current adaptive predictions: `Token2Token/artifacts/threshold_lookahead_v7/full_train7473_t090_num099/full_test_1319/`
- Fixed-k full results: `Token2Token/results/full_k123_1319_raw/`
- Exact fixed-k method notes: `Token2Token/LOOKAHEAD_SELECTION_EXPERIMENT.md`
- Original IG intervention report: `outputs/decode_impact_all/llada-8b_gsm8k/decode_impact_report.md`
- Training configuration and log: `Token2Token/artifacts/threshold_lookahead_v7/full_train7473_t090_num099/train/`
- Research chronology: `Token2Token/RESEARCH_LOG.md`
