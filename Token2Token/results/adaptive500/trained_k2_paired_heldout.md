# Paired comparison

Examples: 150

| Metric | standard_k1 | trained_k2 |
|---|---:|---:|
| Correct | 117 | 110 |
| Accuracy | 78.0% | 73.3% |
| Forwards/example | 128.0 | 64.0 |

## Quality (McNemar, paired)

- Both correct: 102
- Only standard_k1 correct: 15
- Only trained_k2 correct: 8
- Neither correct: 25
- Two-sided exact p-value: 0.2100

## Latency (paired)

- Mean forward change: -64.00 per example
- Bootstrap 95% CI: [-64.00, -64.00]

## Verdict

On these examples, quality is statistically indistinguishable from baseline, and latency improved significantly.
