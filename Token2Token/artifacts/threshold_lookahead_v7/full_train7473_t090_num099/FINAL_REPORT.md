# Threshold-Lookahead Final Report

LLaDA-8B-Instruct on GSM8K test. Both arms use the same adaptive decoder: text tau=.90, numeric tau=.99, and at most two jointly selected catalysts with p2>=.60 and p2/p1>=.85.

## Main result

| Slice | Arm | Correct | Accuracy | Forwards/example | Tokens/forward (batch-1) |
|---|---|---:|---:|---:|---:|
| All test | Base | 908/1319 | 68.84% | 23.49 | 5.449 |
| All test | Trained | 926/1319 | 70.20% | 17.66 | 7.246 |
| Untouched 192-1318 | Base | 771/1127 | 68.41% | 23.40 | 5.470 |
| Untouched 192-1318 | Trained | 785/1127 | 69.65% | 17.63 | 7.260 |

All-test paired churn: 91 base-only correct, 109 trained-only correct; exact McNemar p=0.2292. The paired accuracy delta is +1.36 pp (bootstrap 95% CI -0.76 to +3.49 pp).
Untouched paired churn: 79 base-only correct, 93 trained-only correct; exact McNemar p=0.3216. Its accuracy delta 95% CI is -1.15 to +3.55 pp.

## Latency

| Arm | Summed per-example forwards | Batch-16 iterations | Computed row-forwards | Wall time | Canvas tokens/s |
|---|---:|---:|---:|---:|---:|
| Base | 30,985 | 3,402 | 54,090 | 104.6 min | 26.90 |
| Trained | 23,299 | 2,733 | 43,494 | 84.7 min | 33.24 |

Forward reduction (batch-1): 24.8%. Batch-16 iteration reduction: 19.7%. Measured wall-time reduction: 19.1%.

## Commit mechanism

| Arm | Tokens/cycle | Threshold tokens/cycle | Second catalysts | Second-catalyst rate | Cleanup tokens |
|---|---:|---:|---:|---:|---:|
| Base | 5.449 | 4.179 | 8,360 | 31.9% | 4,749 |
| Trained | 7.246 | 5.875 | 8,643 | 45.6% | 4,331 |

Threshold bursts explain 94.4% of the tokens-per-cycle increase. The adapter also raises second-catalyst acceptance from 31.9% to 45.6%.

## Standard decoder context

| Model | Correct | Accuracy | Tokens/forward | Wall time |
|---|---:|---:|---:|---:|
| base_llada8b_block32_k1 | 995/1319 | 75.44% | 1.000 | 172.9 min |
| base_llada8b_block32_k2 | 963/1319 | 73.01% | 2.000 | 132.1 min |
| base_llada8b_block32_k3 | 869/1319 | 65.88% | 2.909 | 80.0 min |
| Matched adaptive base | 908/1319 | 68.84% | 5.449 | 104.6 min |
| Threshold-lookahead trained | 926/1319 | 70.20% | 7.246 | 84.7 min |

The trained adaptive model Pareto-improves its matched adaptive base. It does not yet dominate standard block decoding: compared with base block-32 k=2 it is faster but lower-accuracy.

## Merged checkpoint gate (test IDs 128-191)

| Checkpoint | Correct | Forwards | Tokens/forward | Wall time |
|---|---:|---:|---:|---:|
| checkpoint-002000_merged | 46/64 | 1,131 | 7.243 | 280.6s |
| checkpoint-004000-merged | 45/64 | 1,116 | 7.341 | 292.3s |
| checkpoint-006000-merged | 46/64 | 1,099 | 7.454 | 315.8s |
| checkpoint-final-merged | 46/64 | 1,115 | 7.347 | 250.7s |

Checkpoint 6000 was promoted because it tied for the best merged accuracy and used the fewest forwards. The ordering is a selection heuristic on 64 examples, not evidence that checkpoint 6000 is intrinsically better than the tied checkpoints.

## Scope

The adapter was trained only on the GSM8K train split. Decoder thresholds and checkpoint selection touched test IDs 0-191, so the 192-1318 result is the primary untouched estimate. Forward metrics describe logical batch-1 efficiency; measured batch-16 wall time is reported separately because variable sequence completion causes padding waste.
