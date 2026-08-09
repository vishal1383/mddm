# Lookahead Selection LoRA Experiment

## Identity

The model currently evaluated as `lookahead_lora_checkpoint4000` is **not
V4a**. It is the later online-lookahead trainer implemented in
`train_online_lookahead.py`, with the target-selection ranking loss enabled.

Its purpose is narrow: make one student forward support the two positions that
frozen base LLaDA would select over two consecutive standard blockwise `k=1`
forwards. The intended inference setting is therefore fixed blockwise `k=2`.

This experiment is also separate from the successful adaptive base decoder.
The adaptive decoder uses no LoRA and commits a dynamic number of tokens at a
0.99 confidence threshold.

## Why k=2 Was Evaluated First

The evaluation order was chosen to test the trained operating point first:

1. The frozen teacher performs two sequential blockwise `k=1` actions.
2. The student sees the canvas before either action.
3. Training asks that one student forward make both teacher positions usable.
4. Fixed `k=2` then commits those two positions from one forward.

Thus, `k=2` is the direct test of the objective and uses 64 forwards for a
128-token completion, versus 128 forwards for `k=1`. Starting with `k=2` was an
evaluation-priority choice, not evidence that the full `k=1` evaluation had
already completed. The full matrix still evaluates `k=1`, `k=2`, and `k=3` for
every model family.

## Training Data

- Base model: `GSAI-ML/LLaDA-8B-Instruct`
- Dataset: all 7,473 GSM8K training records
- Steps: 7,473, exactly one pass through the cached records
- Completion canvas: 128 tokens
- Decode block length: 32 tokens
- Cached source:
  `outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl`

The cache was built with frozen base LLaDA. On each gold-completion canvas, it
considered alphabetic gold catalyst tokens whose base probability was within
0.7 of the most probable candidate. It selected the candidate maximizing:

```text
number of correct positions above 0.95 after placement
- number of correct positions above 0.95 before placement
```

These older gold-catalyst trajectories provide a pool of realistic partial
canvases. They are not used as the final two token labels described below.

## One Training Step

For each training record:

1. Construct four canvases: the fully masked canvas plus up to three randomly
   sampled cached anchor/post-anchor canvases.
2. Disable the LoRA adapter so the teacher is always frozen base LLaDA.
3. On each canvas, let the teacher select the highest-confidence position in
   the leftmost unfinished 32-token block and commit its predicted token.
4. Run the frozen teacher again on that updated canvas and record its second
   selected position and predicted token.
5. Re-enable the LoRA. Give the student the original, pre-action canvas once.
6. Optimize transition CE, target-selection ranking, and preservation KL.

The two targets are the frozen teacher's predicted tokens and positions. They
are **not greedy-IG gold anchors**, and they are not guaranteed to match the
GSM8K gold rationale. This distinction is important: the experiment distils
the base decoder's next two actions rather than teaching the original IG order.

## Exact Highest-Confidence Selection

For a completion canvas `x` and model logits `z[i, v]` at completion position
`i` and vocabulary token `v`:

1. Set `z[i, MASK] = -infinity`, so the mask token cannot be selected.
2. Compute `p[i, v] = softmax_v(z[i, v])` independently at every position.
3. Define `token[i] = argmax_v p[i, v]`.
4. Define `confidence[i] = p[i, token[i]]`.

Only masked positions in the leftmost unfinished 32-token block are eligible.
If `m` is the smallest still-masked completion index, the active block starts
at `b = floor(m / 32) * 32`; eligible positions satisfy
`b <= i < b + 32` and are still masked. All other position scores are replaced
by `-infinity`.

For the frozen teacher's first action:

```text
p1 = argmax_i confidence[i] over eligible i
t1 = token[p1]
```

The teacher writes `t1` at `p1`, recomputes the entire model forward, and then
applies the same rule to obtain `(p2, t2)`. Thus the second action is conditioned
on the first committed token.

At fixed-`k=2` student inference, there is only one forward. PyTorch
`topk(2)` selects the two eligible positions with largest `confidence[i]`, and
the decoder simultaneously writes `token[i]` at both positions. If only one
eligible mask remains, only that position is written. There is no confidence
minimum, token-type filter, IG score, or 0.99 threshold in this fixed-top-k
rule. The implementation uses PyTorch's native `argmax`/`topk` behavior for
exact ties; it adds no separate tie-break rule.

## Paper-Style Algorithms

Let `D = {(q, y)}` be GSM8K prompt/completion pairs, `B` be frozen base
LLaDA, `S_theta` be `B` with trainable LoRA parameters, `M` be the mask token,
`L=128` be the completion length, and `W=32` be the decode block length.

The phrase "anchor IG" is not mathematically accurate for the cache used by
this checkpoint. Original greedy IG ranked a gold insertion by entropy
reduction. The selector below instead measures the increase in *correct
high-confidence positions*. It uses gold tokens to construct offline training
canvases, but no gold information is used at inference.

### Algorithm 1: Offline Gold-Catalyst Canvas Cache

Inputs are frozen model `B`, gold completion `y`, confidence threshold
`tau=0.95`, and candidate probability ratio `rho=0.7`.

```text
x <- [M, ..., M]                         # fully masked gold-length canvas
T <- []                                  # cached trajectory

while an alphabetic gold token remains masked:
    Run B(q, x).
    For each masked position j:
        yhat_j <- argmax_v P_B(v | q, x, j)
        c_j    <- max_v P_B(v | q, x, j)

    A <- alphabetic masked positions i satisfying
         P_B(y_i | q, x, i) >= rho * max_r P_B(y_r | q, x, r)

    for each i in A:
        x_i <- x with gold token y_i inserted at position i
        Run B(q, x_i).

        U_before(i) <- {j != i : x_j=M, yhat_j=y_j, c_j >= tau}
        U_after(i)  <- {j != i : x_j=M,
                        argmax_v P_B(v | q, x_i, j)=y_j,
                        max_v P_B(v | q, x_i, j) >= tau}
        gain(i) <- |U_after(i)| - |U_before(i)|

    i_star <- lexicographic argmax over i of
              (gain(i), |U_after(i)|, log P_B(y_i | q,x,i), -i)

    Cache both x and x with y_i_star inserted.
    x[i_star] <- y_i_star
    For every j in U_after(i_star): x[j] <- y_j
```

The tie-break therefore prefers larger gain, then more correct positions after
placement, then a more probable gold catalyst, then the leftmost position.
The resulting cache contains gold-conditioned partial canvases. These
catalysts are context states only; they are not the CE labels in Algorithm 2.

### Algorithm 2: Two-Step Online Lookahead LoRA

Inputs are frozen teacher `B`, student `S_theta`, and the cache from Algorithm
1. One optimizer step is taken per GSM8K record.

```text
for each (q, y) in D:                    # 7,473 records, one epoch
    X <- {fully masked canvas}
         union up to 3 sampled cached partial canvases

    for each original canvas x in X:
        # Frozen teacher: two sequential standard k=1 actions.
        (p1, t1) <- HighestConfidenceAction(B, q, x, W)
        x1 <- x; x1[p1] <- t1
        (p2, t2) <- HighestConfidenceAction(B, q, x1, W)

        # Student receives the original canvas once; t1 is not inserted.
        z_theta <- S_theta(q, x)

        L_CE(x) <- -log softmax(z_theta[p2, :])[t2]

        s1 <- log softmax(z_theta[p1, :])[t1]
        s2 <- log softmax(z_theta[p2, :])[t2]
        h  <- max over eligible i not in {p1,p2} of
              max_v log softmax(z_theta[i, :])[v]
        L_select(x) <- softplus(h - min(s1, s2) + 0.1)

        Add every KL(P_B(. | q,x,i) || P_theta(. | q,x,i))
        for masked positions i != p2 to a shared list K.

    L <- mean_x L_CE(x)
         + mean_x L_select(x)
         + 5 * mean(K)
    Update only theta using AdamW; clip gradient norm to 1.0.
```

`HighestConfidenceAction` is exactly the four-step probability/position rule
in the previous section. In particular, `t2` is the frozen teacher's predicted
token, **not necessarily the gold token `y[p2]`**. The CE is ordinary
vocabulary cross-entropy at exactly one future position per sampled canvas:

```text
L_CE = -z_theta[p2,t2] + log sum_v exp(z_theta[p2,v])
```

This objective asks the student to expose on the original canvas the action
that base LLaDA would only expose after first committing `(p1,t1)`.

### Algorithm 3: Fixed-k Student Inference

```text
x <- [M, ..., M]                         # 128 completion masks
while x contains M:
    Run S_theta(q, x) once.
    Find the leftmost unfinished 32-token block.
    At every masked position i in that block, compute
        token[i]      <- argmax_v P_theta(v | q,x,i)
        confidence[i] <- max_v P_theta(v | q,x,i)
    Select the k positions with largest confidence[i].
    Simultaneously write token[i] at those positions.
```

The current primary setting is `k=2`. It has no threshold, IG computation,
gold token, policy network, or teacher call at inference.

### Algorithm 4: Earlier Greedy-IG Anchor Trainer (Not This Checkpoint)

For completeness, this is the exact algorithm that actually used greedy
entropy-reduction IG. It produced the earlier anchor-order LoRA, not
`lookahead_lora_checkpoint4000`.

Offline anchor ordering for gold completion `y`:

```text
C <- non-special, non-whitespace gold-token positions in the first 75% of y
x <- [M, ..., M]
S <- empty ordered list
H_before[j] <- entropy(P_B(. | q,x,j))

for r = 1,...,5:
    for each candidate i in C not already selected:
        x_i <- x; x_i[i] <- y_i
        H_after_i[j] <- entropy(P_B(. | q,x_i,j))
        IG(i) <- sum over j not in selected positions union {i} of
                 H_before[j] - H_after_i[j]

    i_r <- argmax_i IG(i)
    append (i_r, y_i_r, IG(i_r)) to S
    x[i_r] <- y_i_r
    H_before <- H_after_i_r
```

The frozen base model computes this order once, and the saved order never
changes while LoRA trains. For equal floating-point scores, the implementation
keeps the first candidate encountered, which is the leftmost remaining
candidate because candidates are position ordered.

For the stored anchors `(i_1,y_i_1),...,(i_R,y_i_R)`, with `R <= 5`, construct
one canvas per anchor rank:

```text
x^(1) <- [M, ..., M]
x^(r) <- x^(1) with gold anchors 1,...,r-1 inserted, for r > 1
x^final <- x^(1) with all R gold anchors inserted
```

Run all `R+1` canvases in one student batch. The anchor CE is:

```text
L_anchor = (1/R) * sum_r
           [-log P_theta(y_i_r | q, x^(r), position=i_r)]
```

On `x^final`, let `J` be positions that remain masked. The completion CE is:

```text
L_sequence = (1/|J|) * sum_{j in J}
             [-log P_theta(y_j | q, x^final, position=j)]
```

The minimal trainer optimizes:

```text
L_IG_anchor = 1.0 * L_anchor + 1.0 * L_sequence
```

Thus the genuine IG trainer uses **gold-token CE at the IG-selected gold
positions**, conditioned on earlier gold anchors. By contrast, Algorithm 2
uses **teacher-prediction CE at the second future decoder position**. They are
different training objectives and should not be merged into one method claim.

## Losses

For teacher actions `a1` and `a2`, the total loss is:

```text
loss = 1.0 * transition_CE
     + 1.0 * selection_ranking
     + 5.0 * preservation_KL
```

`transition_CE` supervises the second action token `a2` at its teacher-selected
position from the original canvas. It does not add CE for `a1`, because `a1`
is already the frozen base model's immediate action.

`selection_ranking` includes both `a1` and `a2`. It applies a softplus ranking
loss that pushes the weaker target score above the strongest competing
position in the current block, with margin 0.1. This term was added because the
earlier 500-example model often predicted the future token correctly at its
position but did not rank that position among the top two positions to commit.

`preservation_KL` matches the student distribution to frozen base LLaDA at all
still-masked positions except the second action position. It therefore includes
the first action position; only the future target `a2` is excluded so its CE
term can change it.

One implementation detail: `max_unlock_tokens=2` appears in the configuration,
but the current cached-canvas construction does not truncate each historical
round's unlocked tokens. In this trainer, that flag does not change the sampled
canvas pool. The actual two-token constraint comes from `lookahead=2`.

## Optimization Configuration

| Setting | Value |
|---|---:|
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| LoRA modules | `q_proj,k_proj,v_proj,attn_out` |
| Trainable parameters | 8,388,608 (0.1045%) |
| Optimizer | AdamW |
| Learning rate | 3e-5 |
| Gradient clipping | 1.0 |
| Precision | BF16 |
| States per example | 4 |
| Checkpoint interval | 500 steps |
| Seed | 42 |

Checkpoint 4000 was selected because it tied for the best first-50 result with
checkpoint 4500, and the earlier checkpoint was chosen conservatively:

| Model/decoder | First 50 accuracy | Forwards/example |
|---|---:|---:|
| Base LLaDA, fixed `k=1` | 37/50 = 74% | 128 |
| Base LLaDA, fixed `k=2` | 33/50 = 66% | 64 |
| Lookahead LoRA checkpoint 4000, fixed `k=2` | 39/50 = 78% | 64 |

The first 50 examples selected the checkpoint, so this is tuning evidence, not
a final result. The complete 1,319-example GSM8K test evaluation is running for
fixed `k=1`, `k=2`, and `k=3` on base LLaDA, this LoRA, and the ordinary GSM8K
LoRA control.

## Log Interpretation

For fixed top-k runs, a line such as:

```text
threshold=0.950 ... tokens_per_forward=2.000
```

does not mean threshold decoding is active. `0.950` is an unused legacy field
in the shared evaluator. `decoder=topk` and `tokens_per_forward=2.000` describe
the actual fixed `k=2` behavior. The separate adaptive decoder uses threshold
0.99 and a dynamic number of tokens per forward.

## Artifacts

Training configuration and checkpoints in the container:

```text
outputs/token2token/online_lookahead_v6/
  k2_select1_train7473_step7473/train/
```

Full evaluation matrix:

```text
outputs/token2token/full_k123_1319/
```

Durable host backup of all Token2Token outputs:

```text
/home/vishalg/Desktop/DhruveshProjectArtifacts/token2token/
```
