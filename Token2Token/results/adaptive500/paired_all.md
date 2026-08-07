# Paired comparison

Examples: 500

| Metric | standard_k1 | adaptive_t099 |
|---|---:|---:|
| Correct | 383 | 385 |
| Accuracy | 76.6% | 77.0% |
| Forwards/example | 128.0 | 71.2 |

## Quality (McNemar, paired)

- Both correct: 381
- Only standard_k1 correct: 2
- Only adaptive_t099 correct: 4
- Neither correct: 113
- Two-sided exact p-value: 0.6875

## Latency (paired)

- Mean forward change: -56.83 per example
- Bootstrap 95% CI: [-58.32, -55.29]

## Verdict

On these examples, quality is statistically indistinguishable from baseline, and latency improved significantly.
