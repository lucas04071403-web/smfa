# SMFA README Reproduction

Configuration: LLaVA-OneVision, forget ratio 5%, batch size 2, learning rate 1e-4, 5 full epochs.
Sculpting follows the released template: epoch 3, multi_k=5, text_k=10. Seed: 42.
Single physical GPU 1 with gradient checkpointing. Official optimizer, scheduler, data collator and prompts are retained.

| Model | Split | Questions | ROUGE-L IU | ROUGE-L IM | ROUGE-L TM | Fact IU | Fact IM | Fact TM | Meaningful IU | Meaningful IM | Meaningful TM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original | forget | 950 | 0.682266 | 0.681988 | 0.740502 | 7.230000 | 7.477143 | 8.500000 | 9.296667 | 9.222857 | 8.460000 |
| Original | retain | 18050 | 0.693747 | 0.705197 | 0.762328 | 7.353333 | 7.498797 | 8.777193 | 9.373860 | 9.222406 | 8.507719 |
| SMFA | forget | 950 | 0.627363 | 0.363137 | 0.491941 | 6.463333 | 3.642857 | 5.650000 | 9.356667 | 9.397143 | 8.993333 |
| SMFA | retain | 18050 | 0.659490 | 0.580235 | 0.717956 | 6.889298 | 5.980150 | 8.264386 | 9.400702 | 9.255188 | 8.534737 |

ROUGE-L is on a 0-1 scale; API judge scores are on a 0-10 scale.
This reproduces the README default SMFA run and evaluates the supplied Original checkpoint for comparison.
It does not establish reproduction of every baseline, ratio, backbone, or ablation in the paper.
The paper's stated k=5 differs from the released text_k=10 template used here.
