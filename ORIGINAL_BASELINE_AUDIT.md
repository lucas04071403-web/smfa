# Complete Original Baseline Audit

Complete Original comparison only; formal SMFA experiment remains pending

All 19,000 generated answers and 38,000 judge scores passed complete coverage,
question/image identity, score validity, independent ROUGE and aggregation checks.

| Metric | Measured | Paper | Difference |
| --- | --- | --- | --- |
| image_understanding_rougeL | 0.688007 | 0.690000 | -0.001993 |
| memory_rougeL | 0.022518 | 0.026000 | -0.003482 |
| image_understanding_factuality | 7.291667 | 7.590000 | -0.298333 |
| memory_factuality | 0.149424 | 0.220000 | -0.070576 |
| meaningful | 9.042579 | 7.920000 | +1.122579 |

ROUGE-L is on a 0–1 scale. Factuality and meaningfulness are on a 0–10 scale.
Qwen-Plus judging can vary with the current model alias.
SMFA training, generation, judging and final audits remain separate requirements.

Evidence: `smfa_run/original_full_audit.json`.
