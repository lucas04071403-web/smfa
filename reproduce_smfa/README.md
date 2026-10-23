# SMFA Local Reproduction

This runs the README default SMFA experiment, with the supplied Original model
as a comparison. It uses complete datasets and five full training epochs.
The validation-only files outside this directory do not contribute to its results.

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate smllmun
export CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python /root/lucas4/reproduce_smfa/run.py --launch
```

Inspect `/root/lucas4/smfa_run/status.json` and the current stage log in that
directory. `train.log` reports epoch and adapter progress. The pipeline is a
separate process and continues after the terminal closes. Launching a second
copy cannot start concurrent work because the runner holds an exclusive lock.

The sequence is training, sculpting, SMFA generation, SMFA API scoring,
Original generation, Original API scoring, and a verified results report.
The report is `/root/lucas4/SMFA_REPRODUCTION.md` after all stages succeed.
Source hashes and commands are recorded in `smfa_run/manifest.json`.

At 2026-10-21 17:29 UTC, an additional GPU-0 worker passed three actual
cross-GPU generation probes: all 57 answers and serialized result hashes
matched the existing GPU-1 outputs exactly, with unchanged CPU/CUDA RNG states.
`smfa_run/parallel_smfa_profiles.py` then began the distant retain suffix in
reverse order, calling the unchanged official evaluator with the exact same
checkpoint, FP16 loader, processor and greedy decoding settings. The existing
generator keeps running and later reuses these verified per-profile caches.
The suffix worker maintains at least 100 profiles of separation and atomically
moves only complete, identity-checked outputs into absent profile directories.
It caps its Torch allocator at 60% of GPU-0 capacity and does not touch the
unrelated GPU context. Its status and log are
`smfa_run/parallel_smfa_profiles_status.json` and
`smfa_run/parallel_smfa_profiles.log`. Count published suffix profiles separately
from the primary generator's contiguous `progress.json` prefix. This is only
generation acceleration; full scoring and final audit are still required.

When GPU 0 became idle on 2026-10-21 at 03:04 UTC, the Original comparison was
started ahead of the sequential pipeline on that GPU. SMFA training continues
on GPU 1. This uses the same generation/scoring scripts and parameters, with an
isolated output directory. See `smfa_run/parallel_original_status.json` and
`smfa_run/parallel_evaluate_Original.log`; its launcher is
`smfa_run/parallel_original.py`. After generation and judging pass the normal
coverage checks, its directory is moved to the expected Original result path
only if that path does not exist. The sequential pipeline can then reuse the
completed profile and score caches and verify them again.

At 2026-10-21 04:04 UTC, judging of the completed Original forget split was also
started while retain generation continues. Its launcher is
`smfa_run/early_original_forget.py`, with progress in
`smfa_run/early_original_forget_status.json` and
`smfa_run/early_original_forget.log`. It invokes the same official judge through
`score.py`, with four workers and unchanged prompts. Scores are first written to
`smfa_run/early_original_forget/`; after both metrics pass full question and
identity checks, the closed cache/result files are published to the Original
directory using links that cannot replace an existing file. The later controller
can reuse the per-question caches. These are Original forget results only;
the complete experiment still requires both splits and both checkpoints.

The same early judge now accepts a fixed completed profile prefix. At
2026-10-21 04:34 UTC, the first 125 Original retain profiles (2,375 questions)
were queued for both metrics:

```bash
python /root/lucas4/smfa_run/early_original_forget.py --launch --split retain --profiles 125
```

Status and logs are `smfa_run/early_original_retain_125_status.json` and
`smfa_run/early_original_retain_125.log`. The input prefix is checked against
the original dataset, including image IDs and image types. Its input and
launcher hashes are recorded in the status file. Outputs are kept in
`smfa_run/early_original_retain_125/`. Only the two closed per-question caches
are published after both metrics pass checks; partial score summaries stay in
that isolated directory. The later complete retain scoring run reuses matching
question digests and computes full-split means from all 18,050 questions.

That 125-profile task completed at 2026-10-21 05:35:01 UTC. Both caches were
published after all 4,750 scores passed independent verification. Exact prefix
means and initial cache hashes are in `smfa_run/partial_original_retain_125.json`.
The full scorer can append to these shared cache files, so cache hashes describe
their initial published contents; isolated prefix score summaries stay fixed.

At 2026-10-21 05:41 UTC, early judging was extended to the first 250 Original
retain profiles (4,750 questions):

```bash
python /root/lucas4/smfa_run/early_original_forget.py --launch --split retain --profiles 250
```

Status and logs are `smfa_run/early_original_retain_250_status.json` and
`smfa_run/early_original_retain_250.log`. The completed first 125 profiles supply
2,375 cached scores per metric, leaving 4,750 new API scores across both metrics.
The early launcher validates and copies matching cache records before scoring.
After both metrics finish, it validates and appends only missing records to the
existing published caches, preserving the prior records and shared files.
An exclusive lock prevents concurrent early judges; publication also checks that
Original generation is still running and is away from the full-scoring handoff.
If that handoff has already occurred, outputs stay isolated for later merging.
Score conflicts are rejected. Partial score summaries remain isolated, and the
complete scorer still calculates means from all 18,050 retain questions.

At 2026-10-21 06:00 UTC, `smfa_run/original_judge_queue.py` was started to wait
for the running 250-profile task, then judge the 375, 500, 625 and 750-profile
prefixes sequentially as their generations become available. Status is in
`smfa_run/original_judge_queue_status.json`; its log is
`smfa_run/original_judge_queue.log`, and each prefix keeps its own usual status,
log and isolated summary directory. There is still only one early judge with
four workers. Each prefix reuses matching existing scores and appends only its
new records after validation. The queue stops starting early tasks when Original
generation hands off to complete scoring. A failed child keeps its saved cache
and stops the queue for inspection; it does not restart training or generation.
The source hashes are recorded at launch. Offline checks verified ordered
execution, waiting for the prior task, no overlapping children, and the handoff
behavior without API calls.

An independent CPU-only final auditor is available at
`smfa_run/audit_final.py`. Its watcher waits for the full controller to report
all six stages complete before inspecting any final results. It checks all 15
adapters, every sculpted tensor shard, exact question and image identities,
both complete judge caches, and independently recomputed ROUGE-L and score
means. It also calculates the paper's overall metrics and numerical differences.
It saves `smfa_run/final_audit.json` and `SMFA_FINAL_AUDIT.md` only after all
checks pass. Its status and log are `smfa_run/final_audit_status.json` and
`smfa_run/final_audit.log`. It never changes training, generation or judge
outputs, and does not issue API requests. The validation record is
`smfa_run/final_audit_validation.json`.

```bash
python /root/lucas4/smfa_run/audit_final.py --launch
```

Two supplemental audit watchers were added at 2026-10-21 10:11 UTC. The formal
trainer, sculptor, generator, scorer, runner and frozen final auditor remain
unchanged.

`smfa_run/audit_sculpt_formula.py` waits for the formal sculpt stage to finish
and for physical GPU 0 to be idle. It then reconstructs every one of the 274
adapted weights from the actual epoch-3 LoRA matrices and Original checkpoint,
using the released thresholds, CPU mask calculation, FP32 deltas and FP16 merge.
It requires exact saved-weight equality and exact aggregate mask counts. Its
status/log/result are `smfa_run/sculpt_formula_audit_status.json`,
`smfa_run/sculpt_formula_audit.log` and `smfa_run/sculpt_formula_audit.json`.
This additional audit never changes checkpoints or evaluation outputs and makes
no API requests; it allocates no GPU memory while waiting.

The actual complete formula audit passed at 2026-10-21 17:16:03 UTC. Rather
than waiting for GPU 0 to have no context at all, a bounded run used its measured
43 GiB of spare capacity with a 4 GiB PyTorch allocator cap (657.625 MiB peak).
The idle formula watcher alone was replaced; training, generation and scoring
were untouched. All 274 saved adapted weights and both aggregate mask totals
match exact independent reconstruction. Evidence and canonical completion are
`smfa_run/sculpt_formula_audit.json` and
`smfa_run/sculpt_formula_audit_status.json`.

The original formula auditor assumed saved FP32 adapter tensors were loaded
unchanged. The installed PEFT loading sequence preserves the first adapter,
but rounds the two later adapters through FP16 before promoting them to FP32.
`smfa_run/audit_sculpt_formula_loader.py` verifies that behavior through an
actual PEFT probe with real layer matrices, then uses the existing independent
formula checker with that loading policy. Every saved weight and mask total
must still match exactly; no tolerance or checkpoint change is permitted.
Failed original-audit evidence, installed PEFT source hashes and the loader
probe are preserved in the correction directory recorded in the audit result.
The frozen experiment scripts and frozen final auditor are unchanged.

Transformers 4.53 converts old LLaVA checkpoint names such as
`language_model.model.layers.*` to runtime names such as
`model.language_model.layers.*`, then reverses that mapping when saving. LoRA
adapter names use the runtime form. The frozen final auditor currently passes
these derived runtime names directly to its saved-checkpoint check, which will
reject them. `smfa_run/audit_final_names.py` waits for the original auditor to
exit naturally after the full pipeline finishes. If it stops on that exact
known name mismatch, the entrypoint backs up its status/log, translates only
the 274 adapted layer names using `smfa_run/checkpoint_names.py`, and runs all
unchanged final checks. Other failures stop it for inspection. Its status and
log are `smfa_run/final_audit_names_status.json` and
`smfa_run/final_audit_names.log`. A successful run writes the usual final audit
and report with additional mapping/source provenance. The original watcher and
all experiment processes were left running; its pinned source is preserved for
the early SMFA queue.

Offline validation confirmed all 765 runtime names map bijectively to the
actual downloaded checkpoint, and every formal epoch-3 LoRA matrix has the
correct corresponding weight shape. Hand-calculated rank-16 cases cover mask
selection, zero deltas and FP16 merging. Disposable synthetic checkpoints
reproduce the frozen audit's name failure, pass with name translation, and
still reject unadapted mutations and nonfinite weights. No GPU allocation, API
request or formal output mutation was used; see
`smfa_run/checkpoint_audits_validation.json` and
`smfa_run/validate_checkpoint_audits.py`. Actual formal sculpted tensors remain
unverified until sculpting finishes and both final audits run.

An early SMFA judge queue can overlap API scoring with SMFA generation:

```bash
python /root/lucas4/smfa_run/early_smfa_queue.py --launch
```

It waits for SMFA generation and for the independent Original controller to
finish judging. It then scores the complete 50-profile forget split and retain
prefixes of 125, 250, 375, 500, 625 and 750 profiles sequentially, using the
existing prompts, validators, four workers and matching score caches. Each
prefix is audited independently before its per-question caches are published;
partial summaries stay isolated. The queue stops launching workers at the full
scoring handoff. A worker finishing after that handoff keeps its output isolated.
Status and logs are `smfa_run/early_smfa_queue_status.json` and
`smfa_run/early_smfa_queue.log`. Failures stop the queue for inspection while
preserving cached scores. The frozen trainer, generator, sculptor, full scorer,
Original helpers and final auditor are unchanged. Offline integration validation
used synthetic scores with real input identities in disposable directories and
made no API calls; see `smfa_run/early_smfa_queue_validation.json` and
`smfa_run/validate_early_smfa_queue.py`.

Closed Original retain prefixes can also be independently inspected without
GPU or API requests. For example, for the completed 500-profile task:

```bash
python /root/lucas4/smfa_run/audit_original_prefix.py --profiles 500 --previous 375
```

Use `--output <new-path>` to record a new audit; existing audit files are never
replaced. The recorded 500-profile audit is
`smfa_run/partial_original_retain_500.json`. The auditor checks
all questions, images, scores, digests, summaries and ROUGE against the dataset
and original profile outputs, verifies reused scores against the previous
completed prefix, and checks its publication-time cache byte hashes. It refuses
running or incomplete tasks. Validation used the completed 375-profile task
(14,250 scores) and malformed-cache cases without API calls. See
`smfa_run/audit_original_prefix_validation.json` and
`smfa_run/verification_original_retain_375.json`. These are partial-prefix checks;
the final auditor still requires complete results for both checkpoints.

The 500-profile Original retain task completed at 2026-10-21 08:51:06 UTC.
Independent auditing verified all 9,500 questions and 19,000 scores, including
unchanged previous 7,125 records and their publication-time byte hashes per
metric. The partial summaries remain isolated. Two meaningfulness questions
(indexes 9397 and 9404) exhausted normal retries because Qwen-Plus returned
illegal JSON apostrophe escapes. Once both the scorer and queue had exited,
`smfa_run/recover_retain_500.py` backed up their failed state and cache, then
requested only those two questions using the official messages and parameters.
Only those invalid escapes were normalized before the unchanged validator ran;
the 9,498 valid meaningfulness records and the factuality cache were preserved
byte for byte. Raw/normalized response evidence and independent verification
are in `smfa_run/retain_500_json_recovery_20261021T085025Z/`. The original helper
then completed and published both caches without further API calls. The queue
resumed with 625 profiles at 08:51 UTC, reusing 9,500 scores per metric. These
are partial Original results, not full retain or SMFA results.

The 625-profile factuality worker and queue stopped at 09:20 UTC with two
missing scores (indexes 9796 and 11348) after normal retries exhausted on the
same illegal JSON apostrophe escape. `smfa_run/recover_judge_cache.py` backed
up the closed cache, statuses and log, then requested only those two questions.
Their returned scores were both 10. Independent verification checked unchanged
official requests, exact captured scores and justifications, and preservation
of all 11,873 valid record bytes. The evidence directory is
`smfa_run/retain_625_factuality_json_recovery_20261021T092115Z/`.
Offline recovery validation used real question identities and a synthetic
completion in a disposable directory, without API calls or changes to formal
inputs; see `smfa_run/recover_judge_cache_validation.json`.

At 09:21 UTC the unchanged early helper resumed with all 11,875 factuality scores
cached, then began meaningfulness with 9,500 reused scores. Both metrics still
need the helper's final checks and cache publication. A CPU-only follow-up waits
for successful completion, audits the 625-profile prefix against the previous
500-profile audit, and resumes the 750-profile queue only while generation is
active. Its state is `smfa_run/original_625_followup_status.json`; failures stop
it for inspection. Neither training nor generation was restarted.

The 625-profile task completed at 09:53:45 UTC. At 09:54 UTC the follow-up
independently audited all 11,875 questions and 23,750 scores, including exact
identity, ROUGE, category means, previous 9,500 scores per metric and their
published byte hashes. Both per-question caches are published; partial summaries
remain isolated. Evidence is `smfa_run/partial_original_retain_625.json`.
Meaningfulness category means are 9.369067, 9.221257 and 8.525867; its
question-weighted mean is 9.048337. These remain prefix-only Original results.
The queue resumed with the 750-profile task (PID 1875073), reusing 11,875
factuality scores. A CPU-only watcher waits to independently audit that closed
prefix against the 625-profile record; see
`smfa_run/original_750_followup_status.json`. It stops on worker failure and
does not request API scores or change experiment outputs.

The 750-profile task completed factuality and entered meaningfulness at about
10:23 UTC. Independent cache/summary verification at 10:24 UTC checked all
14,250 factuality scores, unique indexes, digests, question/answer/image
identities, valid ranges and category means. All 11,875 reused scores and the
prior published byte hash remain unchanged. Category means are 7.366222,
7.483238 and 8.768889. Evidence is
`smfa_run/verification_original_retain_750_factuality.json`. Meaningfulness
reused 11,875 scores and continues with the additional questions. Full prefix
publication and the two-metric independent audit remain pending.

The 750-profile task completed both metrics and cache publication at
10:54:06 UTC. Its CPU follow-up audited all 14,250 questions and 28,500 scores
at 10:54:30 UTC, including independently recomputed ROUGE, category means,
identities, and preservation of the prior 11,875 scores and published record
bytes per metric. Evidence is `smfa_run/partial_original_retain_750.json`.
Meaningfulness category means are 9.373556, 9.215048 and 8.518444; its
question-weighted prefix mean is 9.045123. The early Original queue completed
all its targets and exited. These remain prefix-only comparison results.
Original generation continues toward all 950 retain profiles; its full scorer
will reuse 14,250 retain scores per metric and score the remaining 3,800.

`smfa_run/audit_original_full.py` now waits for the independent Original
controller to complete generation, both judge metrics and result promotion.
It then uses the unchanged split auditor on all 19,000 answers and 38,000
scores, checks the two evaluation datasets against their full-audit hashes,
recomputes ROUGE/category means/overall metrics, and compares with the paper's
Original row. It writes `smfa_run/original_full_audit.json` and
`ORIGINAL_BASELINE_AUDIT.md` only after successful checks. Live status and log
are `smfa_run/original_full_audit_status.json` and
`smfa_run/original_full_audit.log`. This read-only CPU watcher uses no API or
GPU work and stops if the Original controller fails. It is a complete baseline
audit, separate from the eventual full SMFA final audit. Its split-audit logic
is the existing frozen auditor already validated against real closed results.

The remaining-epoch audit watcher started at 2026-10-21 11:19 UTC:

```bash
python /root/lucas4/smfa_run/audit_training_epochs.py --launch
```

It waits until each epoch-4/5 checkpoint has closed, then checks the four full
batch counts (600/500/300/250), finite final losses, all three adapter files,
LoRA configuration, tensor coverage/shapes/finiteness, learned language/vision
B matrices and changes from the preceding audited epoch. It records checkpoint
hashes in `smfa_run/epoch_4_audit.json` and `smfa_run/epoch_5_audit.json`.
Status/log are `smfa_run/training_epoch_audit_status.json` and
`smfa_run/training_epoch_audit.log`. This CPU-only watcher does not change
checkpoints, allocate GPU memory or request API scores. Validation exactly
reproduced the completed epoch-3 audit and rejected incomplete-batch and
nonfinite-loss evidence; see
`smfa_run/training_epoch_auditor_validation.json`. The formal training and
generation processes continue uninterrupted.

Complete Original generation finished at 2026-10-21 11:49 UTC. The controller
verified all 19,000 answers and entered full judging, reusing the validated
14,250-question retain caches. An independent CPU-only generation audit checked
both full datasets, all 1,000 per-profile outputs, question/image identities,
nonempty answers and recomputed ROUGE-L. It passed; evidence is
`smfa_run/original_generation_audit.json`. Original overall generation metrics
are image understanding 0.688007 and memory 0.022518 (paper: 0.690 and 0.026).
API metrics and the complete SMFA experiment remain pending; this generation
audit does not replace the full baseline and final audits described above.

Complete Original factuality has also been independently verified for all
19,000 questions: cache coverage/uniqueness, question/image identities,
digests, integer ranges, justifications and exact category summaries passed.
Evidence is `smfa_run/verification_original_full_factuality.json`.
Overall image understanding is 7.291667 and memory is 0.149424, versus the
paper's 7.59 and 0.22. Full meaningfulness is still running; a malformed JSON
response left index 14268 without a valid score. The running cache is preserved;
closed-worker recovery must retain valid record bytes and official requests.
Neither the complete baseline audit nor the SMFA experiment is complete yet.

Recovery for the complete Original scoring controller is prepared separately
from the early-prefix helpers. Do not run it while the controller or scoring
child is live. Once they have exited, inspect the final invalid-index set and
use the activated Conda environment:

```bash
python /root/lucas4/smfa_run/recover_original_full.py --indexes 14268
python /root/lucas4/smfa_run/resume_original_scoring.py --launch
```

The inspected live failure is currently index 14268; more failures may appear
before the worker closes, so the final index set must be checked first. The
recovery refuses live processes, unexpected coverage, different question
digests, additional invalid records and uninspected missing indexes. It backs
up the closed cache, controller status and logs, calls the unchanged official
judge only for the specified missing scores, permits only illegal JSON
apostrophe escape repairs and saves raw/normalized response evidence. Valid
records remain byte for byte unchanged; only missing records are replaced.

The score-only resumer requires complete valid caches for all 38,000 Original
scores. It runs only the frozen full scorer to produce summaries and promotes
results after the usual checks; it never starts a generation/model process.
Closed controller status/logs and cache hashes are preserved as provenance.
After successful promotion, restart the independent baseline auditor only if
its prior watcher has exited, backing up its failed status/log first. The full
SMFA pipeline remains separate and must continue uninterrupted.

Offline validation rejected the actual live controller and exercised cache
identity, invalid-index, byte-preservation and JSON fault cases. A disposable
integration fixture used all 19,000 real question identities and 38,000
synthetic scores to exercise the frozen cached scorer and atomic promotion,
with no API requests, model loads or GPU allocations. Evidence is
`smfa_run/original_full_recovery_validation.json` and
`smfa_run/original_score_resume_validation.json`. These validations used no
formal output mutations. Formal recovery and promotion subsequently completed
as recorded below.

At 2026-10-21 14:59 UTC, both Original controller processes had exited and the
closed full cache contained exactly one invalid meaningfulness record, index
14268. The recovery requested only that question with the official messages
and model. Its returned score of 4 and justification passed the unchanged
validator after removal of one illegal JSON apostrophe escape. All 18,049
other meaningfulness record bytes were preserved. The independent request
replay, response verification, cache identity checks and protected artifact
hashes are recorded in
`smfa_run/original_full_meaningful_json_recovery_20261021T145935Z/verification.json`.
No additional API call was made by this verification.

The score-only resume reused all 38,000 valid scores and promoted complete
Original results at 14:59:59 UTC. The independent full baseline audit passed
at 15:00:45 UTC, checking all 19,000 answers, both metrics, question/image
identities, score summaries and recomputed ROUGE. See
`smfa_run/original_full_audit.json` and `ORIGINAL_BASELINE_AUDIT.md`.
All five training epochs and formal sculpting have also completed. Full SMFA
generation and scoring and the independent final audits remain required.

The CPU-only `smfa_run/audit_checkpoint_integrity.py` subsequently checked all
15 adapters and all 765 sculpted tensors using the existing frozen audit
functions and verified checkpoint name mapping. All values are finite,
274 adapted weights changed and every unadapted weight matches the Original
checkpoint cast. Saved adapter hashes match the individual epoch audits.
Evidence is `smfa_run/checkpoint_integrity_audit.json`. This checks actual
checkpoint integrity before evaluation closes; the GPU formula reconstruction
and complete final evaluation audit remain separate requirements. No GPU,
API requests, source edits or checkpoint mutations were used by this check.

SMFA forget judging completed and published at 2026-10-21 15:23:29 UTC.
The independent `smfa_run/verification_smfa_forget_50.json` checks all
50 profiles, 950 generated answers and 1,900 factuality/meaningfulness scores
against the dataset, source profile outputs, isolated summaries and published
caches. Every published record byte is preserved by question index, although
publication sorts the records. Independent ROUGE and score means match the
completed prefix audit. The full scorer can reuse both caches; complete SMFA
retain evaluation and the final audits remain required.

Both metrics for the first 125 SMFA retain profiles completed and published
at approximately 2026-10-21 16:31 UTC. The independent
`smfa_run/verification_smfa_retain_125.json` verifies all 2,375 answers and
4,750 scores against the dataset, all profile outputs, isolated summaries
and published cache bytes by index. Previously verified factuality files are
unchanged. The queue then started the 250-profile prefix with 2,375 cached
factuality scores; see `smfa_run/early_smfa_retain_250_status.json`.
These prefixes reduce the later full scoring work; complete retain results
and the final audits remain required.

Local adaptations:

- Gradient checkpointing enables batch size 2 on one A40. FP32 backbone
  preparation, AdamW, the released scheduler stepping, and all adapters remain
  as in the official trainer. Random seed is 42.
- Sculpting applies the same conflict/magnitude formula per layer rather than
  retaining all dense delta matrices at once. It uses epoch 3, multi_k=5 and
  text_k=10, exactly as the released shell template specifies.
- Evaluation calls the released generator for each profile and keeps per-profile
  results for resuming. It preserves images, prompts, token limits, decoding,
  sample order and ROUGE scoring. Forget has 950 questions; retain has 18,050.
- API judging uses the released Qwen-Plus prompts and validators, with four
  workers, saved per-question results and checks for absent scores. Both splits
  are scored explicitly because the official filename filter misses retain.
  Three Original retain meaningfulness responses (index 5616 in the 375-profile
  task, and indexes 9397 and 9404 in the 500-profile task) required repair of
  illegal JSON apostrophe escapes after normal retries failed. Replacement
  requests kept the original messages and model parameters; returned scores
  and justifications were preserved and passed the unchanged official
  validator. All 7,124 other records in the 375-profile task and all 9,498 other
  records in the 500-profile task were preserved byte for byte. Raw/normalized
  response provenance for the first recovery is recorded in
  `smfa_run/retain_375_json_recovery_20261021T074510Z/recovery.json`.
  The original helper completed the 375-profile prefix at 07:45:41 UTC on
  2026-10-21, and the queue resumed with the 500-profile prefix. No frozen
  experiment sources were changed and training/generation were not restarted.
  The second recovery and the two factuality responses in the 625-profile task
  are documented above. Both 625-profile metrics and publication passed the
  independent audit at 09:54 UTC. Both 750-profile metrics and publication
  passed the independent audit at 10:54 UTC without additional recovery.
- Dataset/model downloads and optional fine-tuning are already satisfied by
  the supplied checkpoints. No additional base checkpoint fine-tuning is run.

A successful script exit alone is insufficient: the pipeline verifies all 15
adapter files, sculpted checkpoint shards, complete question coverage and all
judge scores. Training has epoch output checkpoints but the released trainer
does not save optimizer/RNG state for exact interrupted-step continuation.
Generation and API scoring can resume from completed profiles/questions.

These settings reproduce the README default experiment. They do not claim to
reproduce all eight baselines, all three forget ratios, both backbones or all
paper ablations. The paper states k=5, whereas the released template uses
text_k=10; this run follows the template.
