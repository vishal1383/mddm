# Paired comparison

Examples: 450

| Metric | standard_k1 | adaptive_t099 |
|---|---:|---:|
| Correct | 346 | 349 |
| Accuracy | 76.9% | 77.6% |
| Forwards/example | 128.0 | 70.9 |

## Quality (McNemar, paired)

- Both correct: 345
- Only standard_k1 correct: 1
- Only adaptive_t099 correct: 4
- Neither correct: 100
- Two-sided exact p-value: 0.3750

## Latency (paired)

- Mean forward change: -57.08 per example
- Bootstrap 95% CI: [-58.65, -55.47]

## Verdict

On these examples, quality is statistically indistinguishable from baseline, and latency improved significantly.
