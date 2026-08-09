# Exact Algorithm for checkpoint-004000

This file describes only the model evaluated as
**lookahead_lora_checkpoint4000**.

The accurate one-line description is:

> Gold-anchor-cache lookahead plus two-step decoder-lookahead distillation.

The training pipeline uses two different forms of lookahead:

1. **Cache lookahead:** try placing each plausible gold anchor and measure how
   many other gold tokens become correct with at least 95% confidence.
2. **Decoder lookahead:** from each blank or cached canvas, run frozen base
   LLaDA for two sequential confidence-decoding actions.

The 95% threshold and gold anchors are part of checkpoint4000 training because
they determine the cached canvases used by the trainer. The final fixed-k=2
inference rule does not use that threshold or gold information.

## Stage A: Construct the Gold-Anchor Cache

The cache is:

~~~text
outputs/token2token/threshold_unlock/
  gsm8k_train_t095_gain_text_q07_max512.jsonl
~~~

It was generated once using frozen GSAI-ML/LLaDA-8B-Instruct on all 7,473
GSM8K training records.

### Cache parameters

~~~text
confidence threshold tau = 0.95
candidate probability ratio = 0.7
candidate filter = alphabetic gold tokens
selection scope = full completion canvas
maximum cached completion length = 512
~~~

For prompt q, gold completion tokens y[1:L], and current partial canvas x, one
cache round does the following.

### A1. Find plausible gold-anchor candidates

A masked position i is a catalyst-anchor candidate only if its gold token:

- is not a special token; and
- decodes to nonempty alphabetic text after stripping whitespace.

Numbers, punctuation, and whitespace cannot be selected as the catalyst
anchor. They can still be inserted later as unlocked tokens.

For every candidate i, frozen base LLaDA computes:

~~~text
anchor_probability(i) = P_base(y[i] at position i | q, x)
~~~

This is the probability of the gold token specifically. The gold token does
not need to be the model's top-1 token at that position.

Keep only relatively plausible candidates:

~~~text
anchor_probability(i) >= 0.7 * max_j anchor_probability(j)
~~~

The 0.7 value is relative to the best eligible gold token. It is not an
absolute 70% probability threshold.

### A2. Measure each candidate's unlocked tokens

Before placing candidate i, define:

~~~text
U_before(i) = {
    j != i:
    x[j] is masked,
    base top-1 token at j equals gold y[j],
    base top-1 confidence at j >= 0.95
}
~~~

Temporarily reveal the candidate's gold token:

~~~text
x_i = copy(x)
x_i[i] = y[i]
~~~

Run frozen base LLaDA again and define:

~~~text
U_after(i) = {
    j != i:
    x_i[j] is masked,
    base top-1 token at j equals gold y[j],
    base top-1 confidence at j >= 0.95
}
~~~

The primary score is:

~~~text
unlock_gain(i) = |U_after(i)| - |U_before(i)|
~~~

Candidates are ranked lexicographically by:

~~~text
(
    unlock_gain(i),
    |U_after(i)|,
    log anchor_probability(i),
    -i
)
~~~

Therefore, **yes: the selected anchor is the plausible gold text token that
creates the largest number of additional gold-correct, at-least-95%-confident
positions**.

Ties prefer:

1. More total correct 95% positions after placement.
2. A more probable gold anchor token.
3. The leftmost position.

The code calls the positions in U_after **unlocked tokens**. These are the
cache's lookahead tokens. They are different from the two model-generated
teacher actions in Stage C.

### A3. Permanently update the cached canvas

After choosing i-star, insert:

~~~text
x[i-star] = y[i-star]                 # selected gold catalyst
x[j] = y[j] for every j in U_after   # unlocked gold tokens
~~~

All positions in U_after are inserted, including positions that were already
correct before the intervention. The update is not limited to only the newly
gained subset.

Unlocked tokens can be numbers or punctuation because the alphabetic filter
applies only to the catalyst candidate.

Repeat Stage A until no alphabetic gold catalyst remains.

## Stage B: Reuse the Cache During checkpoint4000 Training

For every GSM8K training record, the trainer creates up to four 128-token
starting canvases:

~~~text
one fully masked canvas
+ up to three randomly sampled cached partial canvases
~~~

The cache pool contains:

- the canvas before each selected gold catalyst is inserted; and
- the canvas just after that gold catalyst is inserted.

Canvases from later cache rounds also contain all anchors and unlocked gold
tokens inserted in earlier rounds.

Therefore, checkpoint4000 directly reuses:

- the gold anchors;
- the intervention lookahead used to select them; and
- the 0.95 rule used to determine their unlocked tokens.

Those mechanisms shape the model's training inputs even though they are not
rerun online during LoRA optimization.

The run configuration contains max_unlock_tokens=2, but
anchor_transitions(...) does not cap how many cached unlocked tokens appear
in a sampled canvas. The actual two-action setting comes from lookahead=2.

The run was configured for 7,473 examples and 7,473 optimization steps.
checkpoint-004000 is the intermediate snapshot after step 4,000, so that
snapshot had processed the first 4,000 records once.

## Stage C: Select Two Frozen-Teacher Predictions

For every starting canvas x from Stage B:

1. Disable the LoRA adapter.
2. Use frozen base LLaDA as the teacher.
3. Restrict eligible masks to the leftmost unfinished 32-token block.

At every eligible position i, compute:

~~~text
predicted_token(i) = argmax_v P_base(v at i | q, x)
confidence(i) = max_v P_base(v at i | q, x)
~~~

Select the first teacher action:

~~~text
p1 = argmax_i confidence(i)
t1 = predicted_token(p1)
~~~

Insert that model-generated prediction:

~~~text
x1 = copy(x)
x1[p1] = t1
~~~

Run frozen base LLaDA again on x1 and select:

~~~text
p2 = highest-confidence remaining position in the active block
t2 = teacher top-1 token at p2
~~~

For these two actions:

- there is no gold-correctness check;
- there is no 0.95 threshold;
- there is no alphabetic-token filter; and
- either prediction can be wrong relative to the GSM8K gold completion.

This is the second lookahead mechanism: p2 and t2 are what ordinary
confidence k=1 decoding would choose one model forward after p1 and t1.

## Stage D: Train the LoRA Student

The student receives the original starting canvas x. The first teacher action
t1 is not inserted into the student's input.

The objective is:

~~~text
loss = transition_CE + selection_loss + 5 * preservation_KL
~~~

### Transition CE

With lookahead=2, CE is applied only to the second teacher action:

~~~text
transition_CE = -log P_student(t2 at p2 | q, x)
~~~

t2 is the frozen teacher's generated token. It is not necessarily the gold
token y[p2].

### Selection loss

The selection loss pushes both exact teacher actions, (p1,t1) and (p2,t2),
above every competing masked position in the active block. It compares the
weaker target score to the strongest competitor using margin 0.1.

### Preservation KL

KL keeps the student's original-canvas distribution close to frozen base
LLaDA at the other masked positions. It excludes p2, where transition CE
teaches the future action, but includes p1.

Only LoRA parameters are trained:

~~~text
rank = 8
alpha = 16
dropout = 0.05
targets = q_proj,k_proj,v_proj,attn_out
learning rate = 3e-5
~~~

## Stage E: Fixed-k=2 Inference

Inference starts from a fully masked 128-token completion. It does not load
the gold cache.

On each forward:

1. Find the leftmost unfinished 32-token block.
2. Compute each masked position's top-1 token and confidence.
3. Select the two positions with the largest confidences.
4. Commit both model-generated tokens simultaneously.
5. Repeat until the canvas is complete.

Inference uses:

- no gold information;
- no 0.95 threshold;
- no 0.7 plausibility filter;
- no alphabetic-token filter;
- no explicit unlock-gain computation; and
- no teacher model.

The threshold=0.950 text printed by the evaluator is unused metadata under
decoder=topk. Fixed k=2 commits two predictions regardless of their absolute
confidence, except when only one mask remains.

For a 128-token completion, fixed k=2 uses 64 model forwards instead of the
128 forwards used by fixed k=1.

## Exact Use of Gold and the 0.95 Threshold

| Pipeline operation | Uses gold? | Uses 0.95? |
|---|---:|---:|
| Score candidate cache anchors | Yes | Yes |
| Select cached unlocked tokens | Yes | Yes |
| Construct anchor-filled training canvases | Yes | Yes, through the cache |
| Select online teacher action 1 | No | No |
| Select online teacher action 2 | No | No |
| Form the transition CE label | No; teacher-generated | No |
| Fixed-k=2 inference | No | No |

The precise statement is:

> checkpoint4000 is trained on gold-anchor states selected by intervention
> lookahead and the 0.95 unlock rule. It then learns a separate two-action
> decoder lookahead. Only its fixed-k=2 inference is threshold-free.

## Full GSM8K Test Result

| Model | Decode | Accuracy | Forwards/example |
|---|---:|---:|---:|
| Base LLaDA | k=1 | 75.44% | 128 |
| Base LLaDA | k=2 | 73.01% | 64 |
| checkpoint4000 | k=2 | 74.07% | 64 |

checkpoint4000 improves over naive base-model k=2 by 1.06 percentage points
at the same forward count. It remains 1.36 points below base-model k=1 while
using half as many model forwards.
