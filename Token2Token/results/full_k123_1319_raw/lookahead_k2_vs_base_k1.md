# Paired comparison

Examples: 1319

| Metric | base_k1 | lookahead_lora_k2 |
|---|---:|---:|
| Correct | 995 | 977 |
| Accuracy | 75.4% | 74.1% |
| Forwards/example | 128.0 | 64.0 |

## Quality (McNemar, paired)

- Both correct: 902
- Only base_k1 correct: 93
- Only lookahead_lora_k2 correct: 75
- Neither correct: 249
- Two-sided exact p-value: 0.1895

## Latency (paired)

- Mean forward change: -64.00 per example
- Bootstrap 95% CI: [-64.00, -64.00]

## Verdict

On these examples, quality is statistically indistinguishable from baseline, and latency improved significantly.
