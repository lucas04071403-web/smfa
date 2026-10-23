# SMFA Reproduction Readiness

Checked on 2026-10-21 UTC against `04_SMFA.pdf` and official repository
commit `01de31254c947419af448fda8668a088c8cf4310`.

## Latest Recheck: 2026-10-21 11:21 UTC

All data, downloaded checkpoints, Conda dependencies and Qwen-Plus remain
prepared. All 96 pinned package versions still match, `pip check` passes,
frozen experiment sources match their launch hashes and the official Git
worktree is clean. The most recent complete asset-hash verification is
`smfa_run/asset_hash_recheck_20261021T0927Z.json` (75 model files and 13 datasets).

Three complete epochs and their nine saved adapters have passed independent
checks; epoch 4 RA Multi is running. Original retain generation has passed
916/950 profiles, in addition to all 950 forget answers. Independent audits
cover both forget metrics and the completed 750-profile retain prefix, totaling
30,400 scores. Full Original judging and all formal SMFA results remain pending.

An additional validated CPU-only watcher now waits to audit the closed
epoch-4/5 adapters and full training-stream evidence. Live status is
`smfa_run/training_epoch_audit_status.json`. The updated readiness snapshot is
`smfa_run/readiness_update_20261021T1121Z.json`; live experiment progress remains
in the two controller status files. Preparation is complete; the full experiment
has not yet finished.

## Earlier Recheck: 2026-10-21 09:34 UTC

Data, all four downloaded model checkpoints, the Conda environment and Qwen-Plus
remain prepared. The 96 pinned dependencies were rechecked in
`/root/anaconda3/envs/smllmun`; all match, and `pip check` passes. Frozen experiment
sources still match their manifest, and the official Git worktree is clean.

Fresh complete hash verification finished at 09:32 UTC: all 75 downloaded model
files (81,400,554,005 bytes), all 2,988 tensor-to-shard mappings and all 13 data
files match their manifests. Evidence is
`smfa_run/asset_hash_recheck_20261021T0927Z.json`.

Three complete training epochs and all nine saved adapters passed independent
checks; epoch 4 is running. Original generation has completed 750 of 950 retain
profiles (14,250 retain answers), in addition to all 950 forget answers. Both
forget metrics and both metrics for the first 500 retain profiles have passed
independent audits. The 625-profile early judge is running. Its two missing
factuality scores were recovered after the worker and queue stopped, preserving
all 11,873 valid record bytes. All 11,875 factuality scores passed independent
checks, and meaningfulness has 10,394 valid scores so far.
Full retain and SMFA metrics, sculpting and the final audit remain pending.

The new snapshot is `smfa_run/readiness_update_20261021T0934Z.json`; dependency
versions are in `smfa_run/readiness_update_20261021T0919Z.json`. Earlier full data and
checkpoint checks remain in `smfa_run/data_audit.json` and
`smfa_run/readiness_recheck.json`. Current experiment status is in
`smfa_run/status.json` and `SMFA_REPRODUCTION.md`.

## Earlier Recheck: 2026-10-21 06:16 UTC

Preparation is complete; the formal experiment is running and final results are
still pending. This update supersedes the initial conditions recorded below.

- Conda `/root/anaconda3/envs/smllmun`, Python 3.10.22: all 96 official dependency
  pins match, and `pip check` reports no broken requirements.
- All four downloaded checkpoints passed a fresh file-size and complete
  tensor-to-shard index check (75 files, 2,988 tensors, 81,400,554,005 bytes).
- The full data audit covers all 13 Parquet files, 9,200 decoded image occurrences
  and 93,200 question/answer occurrences. A fresh SHA-256 check matches all 13
  audited files; see `smfa_run/data_audit.json`.
- Both running controllers and their children have the configured judge
  credential. Conda activation loads it successfully, and the credential file
  has permission 0600. The successful official judge schema probe is saved in
  `smfa_run/judge_probe.json`; no credential value is included in these reports.
- SMFA training PID 1696019 is live on GPU 1. Epoch 1 is complete and all three
  adapters passed finite-tensor and learned-weight checks. Epoch 2 RA Multi is
  at 143/300 batches. Training requires five complete epochs.
- Original generation PID 1721360 is live on GPU 0. All 950 forget answers passed
  question-order, ground-truth and nonempty-output checks, and both sets of 950
  judge scores are complete. Retain has completed 344 of 950 profiles (6,536 of
  18,050 questions), for 7,486 of 19,000 generated answers overall.
- For the first 250 retain profiles, factuality has all 4,750 scores and
  meaningfulness has 2,867 of 4,750 scores. Current records passed score-range
  and question/answer identity checks. These are partial Original results;
  complete retain and SMFA evaluations remain pending.
- All six training, generation, controller and judge/queue processes are live.
  Current logs show no traceback, CUDA error or nonfinite training loss.
- Experiment helper hashes still match the launch manifest, and the official
  Git worktree is clean. Live stage status is in `smfa_run/status.json` and
  `smfa_run/parallel_original_status.json`; the latest check is recorded in
  `smfa_run/readiness_recheck.json`.

## Initial Preparation Record

Subsequent reproduction run: the full README default SMFA pipeline was launched
on 2026-10-21 UTC in Conda `smllmun`, on physical GPU 1. It enables gradient
checkpointing and follows the released epoch-3, multi_k=5, text_k=10 template.
Live status is in `smfa_run/status.json`; see `SMFA_REPRODUCTION.md` and
`reproduce_smfa/README.md`. The readiness checks below are the earlier snapshot.

Local data, weights, Conda dependencies, and the Qwen-Plus API connection are
prepared. Full paper reproduction still requires a training memory configuration
and an explicit choice of the sculpting hyperparameters described below.

## Prepared Assets

| Component | Verification |
| --- | --- |
| Dataset | All 13 expected Parquet files read successfully; columns, sample JSON, and embedded image decoding checked. Includes 1,000 fine-tuning profiles and all 5%, 10%, 15% train/evaluation splits. |
| Models | Both LLaVA-OneVision and Qwen2.5-VL base and original checkpoints present, approximately 81.4 GB in total. All manifest file sizes, configurations, processors, and tensor-to-shard indexes checked. Download-time hashes are recorded in `S-MLLMUn/model/DOWNLOAD_MANIFEST.json`. |
| Environment | Conda prefix `/root/anaconda3/envs/smllmun`; Python 3.10.22. Every version in the official requirements matches, and `pip check` passes. |
| Core packages | PyTorch 2.8.0+cu128, Transformers 4.53.0, PEFT 0.12.0, Accelerate 0.33.0, Datasets 2.21.0. |
| Entrypoints | Imports pass for all 14 training, unlearning, and evaluation Python entrypoints. |
| CUDA | CUDA 12.8 runtime works with the installed NVIDIA driver; matrix multiplication and backward pass verified. |
| Qwen-Plus API | Configured through a root-only credential file outside the repository and Conda activation hooks. A minimal request passed on 2026-10-21 UTC, using 14 tokens. Both judge scripts initialize their clients successfully. |

## GPU Smoke Checks

Tests used physical GPU 1, an otherwise idle A40. GPU 0 was already running another
job. The machine has two A40 GPUs with approximately 45 GiB usable memory each.

The initial single-GPU LLaVA LoRA check exhausted memory during its forward pass.
The official `prepare_model_for_kbit_training` casts the nonquantized backbone to
FP32, making its training memory requirement larger than FP16 inference.
The original failure is preserved in `SMFA_READINESS_default.json` and
`smfa_readiness_check.log`.

A retry with gradient checkpointing enabled passed for both original models:

| Model | Generation | Multimodal LoRA backward | Peak allocated GPU memory |
| --- | --- | --- | --- |
| LLaVA-OneVision original | Passed, 2 new tokens | Finite loss and gradients, 1 sample | 36.9 GB |
| Qwen2.5-VL original | Passed, 1 new token | Finite loss and gradients, 1 sample | 34.3 GB |

These checks use the official multimodal data collator and rank-16 LoRA target
modules. They do not run an optimizer step, all three SMFA adapters together,
full datasets, full epochs, batch size 2, sculpting, or the baseline experiments.
The original training scripts have not been changed.

## Remaining Conditions

1. The official shell templates contain GPU IDs up to 5; choose the actual
   available devices before training. For a single A40, enable gradient
   checkpointing and validate the intended batch size. The tested addition after
   PEFT model preparation is:

   ```python
   model.gradient_checkpointing_enable(
       gradient_checkpointing_kwargs={"use_reentrant": False}
   )
   model.config.use_cache = False
   ```

2. Fact Score and Meaningful Score use `qwen-plus` at the DashScope endpoint.
   Activating `/root/anaconda3/envs/smllmun` now loads `DASHSCOPE_API_KEY`
   automatically; deactivating the environment restores its previous value.
   The credential file is `/root/.config/smfa/credentials.env` (mode 0600).
   Activation also temporarily unsets `ALL_PROXY` and `all_proxy`, whose
   `socks5h` scheme is incompatible with the pinned HTTP client. HTTP/HTTPS
   proxy settings remain available, and deactivation restores both variables.
   The minimal connection check passed; complete metric scoring has not run.
   Sanitized API verification results are in `SMFA_API_CHECK.json`.

3. The paper states `k=5`; the official SMFA template uses `multi_k=5` and
   `text_k=10`. Decide whether the target run follows the paper or the released
   template and record the choice before comparing results.

4. Unlearned models and evaluation results are generated experiment outputs;
   they have not been produced. Full training and metric reproduction remain
   unverified.

## Activate and Recheck

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate /root/anaconda3/envs/smllmun
cd /root/lucas4/S-MLLMUn
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false python ../check_smfa_ready.py --smoke --gradient-checkpointing
```

The GPU recheck temporarily uses substantial GPU memory. Current machine-readable
results are in `SMFA_READINESS.json`; the successful run log is
`smfa_readiness_checkpointing.log`. The exported Conda environment is
`smfa_environment.yml`.
