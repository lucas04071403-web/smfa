# SMFA Final Independent Audit

README default LLaVA-OneVision SMFA, 5% split, five epochs, epoch-3 sculpting; Original comparison

All 15 saved adapters, sculpted checkpoint shards, 38,000 generated answers and
76,000 judge scores passed coverage, identity, range, finiteness and aggregation checks.

| Model | Metric | Measured | Paper | Difference |
| --- | --- | --- | --- | --- |
| Original | image_understanding_rougeL | 0.688007 | 0.690000 | -0.001993 |
| Original | memory_rougeL | 0.022518 | 0.026000 | -0.003482 |
| Original | image_understanding_factuality | 7.291667 | 7.590000 | -0.298333 |
| Original | memory_factuality | 0.149424 | 0.220000 | -0.070576 |
| Original | meaningful | 9.042579 | 7.920000 | +1.122579 |
| SMFA | image_understanding_rougeL | 0.643426 | 0.667000 | -0.023574 |
| SMFA | memory_rougeL | 0.221556 | 0.199000 | +0.022556 |
| SMFA | image_understanding_factuality | 6.676316 | 7.170000 | -0.493684 |
| SMFA | memory_factuality | 2.475840 | 2.300000 | +0.175840 |
| SMFA | meaningful | 9.082789 | 7.980000 | +1.102789 |

Overall image understanding averages forget and retain category means.
Memory is (retain IM + retain TM - forget IM - forget TM) / 2.
Meaningfulness averages all question scores across both complete splits.

- The paper states k=5, while this local run follows the released shell template: multi_k=5, text_k=10, epoch_3.
- Published table values are rounded independently; aggregates calculated from the rounded values can differ by 0.001 or 0.01.
- Qwen-Plus judging can vary over time; the local API scoring uses the released prompts and model alias.

Passing this audit verifies execution and result integrity; the table records numerical
agreement or differences with the paper. This does not reproduce its other baselines or ablations.
Machine-readable evidence: `smfa_run/final_audit.json`.
