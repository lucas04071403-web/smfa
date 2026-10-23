"""Validate identical greedy generation before filling a distant retain suffix."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "smfa_run"
REPO = ROOT / "S-MLLMUn"
DEST = REPO / "eval_result/llava-onevision/split_ratio_5/SMFA"
WORK = RUN / "parallel_smfa_profiles"
STATUS = RUN / "parallel_smfa_profiles_status.json"
GAP = 100
sys.path.insert(0, str(ROOT / "reproduce_smfa"))
import run as runner


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def live(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def verify_sources():
    manifest = read(RUN / "manifest.json")
    if sys.prefix != manifest["conda_prefix"]:
        raise RuntimeError("The recorded Conda environment is required")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    return manifest


def progress():
    pipeline = read(RUN / "status.json")
    if pipeline["state"] != "running" or pipeline["stage"] != "evaluate_SMFA":
        return None
    if not live(pipeline["pid"]) or not live(pipeline["child_pid"]):
        raise RuntimeError("The main controller or generator stopped")
    current = read(DEST / "progress.json")
    if current["split"] != "retain":
        raise RuntimeError("Expected active retain generation")
    return current["completed_profiles"]


def check_questions(path, row):
    from evaluate import verify_questions

    questions = read(path)["Generation_Questions"]
    verify_questions(questions, row)
    original_row = row.iloc[0]
    expected = [(kind, item) for kind, column in (
        ("image_understanding", "image understanding"),
        ("image_memory", "image memory"),
        ("text_memory", "text memory")) for item in original_row[column]]
    for question, (kind, item) in zip(questions, expected):
        answer = question.get("generated_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("An answer is empty")
        if kind != "text_memory" and (question.get("image_id"), question.get("image_type")) != (
                original_row["ID"], item["Image"]):
            raise RuntimeError("Image identity differs")
    return questions


def main():
    import pandas as pd
    import torch
    from evaluate import official

    with (RUN / "parallel_smfa_profiles.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_sources()
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("This worker must be isolated on physical GPU 0")
        if STATUS.exists():
            raise RuntimeError("Inspect the existing suffix worker before reusing its outputs")
        if progress() is None:
            raise RuntimeError("The main generator is unavailable")
        state = {"state": "running", "pid": os.getpid(), "stage": "loading",
                 "started_at": runner.now(), "physical_gpu": "0", "minimum_profile_gap": GAP,
                 "source_sha256": manifest["source_sha256"], "entrypoint_sha256": sha(Path(__file__)),
                 "official_evaluator_sha256": sha(REPO / "eval.py"), "published_profiles": [],
                 "probes": [], "generation": {"max_new_tokens": 100, "do_sample": False},
                 "api_calls": 0, "scope": "Retain suffix acceleration only; full experiment remains pending"}
        runner.write_json(STATUS, state)
        try:
            os.chdir(REPO)
            torch.cuda.set_per_process_memory_fraction(0.60, 0)
            model_dir = REPO / runner.MODEL / "sculpted"
            before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in model_dir.iterdir() if p.is_file()}
            processor = official.AutoProcessor.from_pretrained(runner.MODEL_ID)
            tokenizer = official.AutoTokenizer.from_pretrained(runner.MODEL_ID)
            model = official.LlavaOnevisionForConditionalGeneration.from_pretrained(
                str(model_dir), torch_dtype=torch.float16, device_map="auto",
                low_cpu_mem_usage=True, local_files_only=True)
            if any(module.training for module in model.modules()):
                raise RuntimeError("Model or submodule is in training mode")
            if any(str(parameter.device) != "cuda:0" for parameter in model.parameters()):
                raise RuntimeError("The checkpoint was not loaded entirely on the designated GPU")
            data_path = REPO / "data/S-MLLMUn-data/eval_data/retain_5/train-00000-of-00001.parquet"
            rows = pd.read_parquet(data_path)
            state.update(data_sha256=sha(data_path), total_profiles=len(rows), model_files_before=before,
                         torch=torch.__version__, gpu_name=torch.cuda.get_device_name(0))
            runner.write_json(STATUS, state)

            def generate(index, folder):
                folder.mkdir(parents=True, exist_ok=False)
                row = rows.iloc[index:index + 1]
                parquet = folder / "input.parquet"
                row.to_parquet(parquet, index=False)
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state(0).clone()
                official.evaluate_generation(str(parquet), processor, tokenizer, model,
                                             SimpleNamespace(output_folder=str(folder)), "retain")
                if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state(0)):
                    raise RuntimeError("Profile generation changed a Torch RNG state")
                result = folder / "retain_generation_results.json"
                questions = check_questions(result, row)
                parquet.unlink()
                return result, questions

            state["stage"] = "exact_output_probes"
            runner.write_json(STATUS, state)
            for index in (0, 125, 250):
                baseline = DEST / ".profiles/retain" / f"{index:04d}" / "retain_generation_results.json"
                baseline_hash = sha(baseline)
                result, questions = generate(index, WORK / "probes" / f"{index:04d}")
                if questions != check_questions(baseline, rows.iloc[index:index + 1]):
                    raise RuntimeError(f"Cross-GPU generation differs at profile {index}; no suffix will be published")
                if sha(baseline) != baseline_hash:
                    raise RuntimeError("Probe baseline changed")
                state["probes"].append({"profile": index, "questions": len(questions), "exact_output_equal": True,
                                        "cpu_cuda_rng_unchanged": True, "baseline_sha256": baseline_hash,
                                        "probe_sha256": sha(result)})
                runner.write_json(STATUS, state)
            state["stage"] = "generating_suffix"
            runner.write_json(STATUS, state)

            # Reverse traversal keeps this worker distant from the ascending main process.
            for index in range(len(rows) - 1, -1, -1):
                current = progress()
                if current is None or index - current < GAP:
                    state["stop_reason"] = "main_handoff_or_minimum_gap"
                    break
                verify_sources()
                target = DEST / ".profiles/retain" / f"{index:04d}"
                if target.exists():
                    raise RuntimeError(f"A distant target already exists: {index}")
                staged = WORK / "retain" / f"{index:04d}"
                result, questions = generate(index, staged)
                current = progress()
                if current is None or index - current < GAP:
                    state["unpublished_profile"] = index
                    state["stop_reason"] = "minimum_gap_before_publication"
                    break
                if target.exists():
                    raise RuntimeError(f"The main generator reached a suffix target: {index}")
                record = {"profile": index, "questions": len(questions), "sha256": sha(result),
                          "main_completed_profiles_at_publication": current, "published_at": runner.now(),
                          "cpu_cuda_rng_unchanged": True}
                # A completed directory is moved atomically, with a large ungenerated gap.
                staged.rename(target)
                if sha(target / result.name) != record["sha256"]:
                    raise RuntimeError("Published generation bytes differ")
                check_questions(target / result.name, rows.iloc[index:index + 1])
                state["published_profiles"].append(record)
                state.update(published_questions=sum(p["questions"] for p in state["published_profiles"]),
                             checked_at=runner.now(), main_completed_profiles=current,
                             max_torch_memory_allocated_bytes=torch.cuda.max_memory_allocated(0))
                runner.write_json(STATUS, state)
                print(f"PUBLISHED retain profile {index}; suffix count={len(state['published_profiles'])}; main={current}", flush=True)
            after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in model_dir.iterdir() if p.is_file()}
            if after != before:
                raise RuntimeError("Checkpoint file metadata changed")
            verify_sources()
            state.update(state="complete", stage="complete", completed_at=runner.now(), checkpoint_metadata_unchanged=True)
            runner.write_json(STATUS, state)
        except BaseException as error:
            state.update(state="failed", error=str(error), failed_at=runner.now())
            runner.write_json(STATUS, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        pipeline = read(RUN / "status.json")
        raw = Path(f"/proc/{pipeline['pid']}/environ").read_bytes()
        environment = {os.fsdecode(key): os.fsdecode(value) for key, value in
                       (item.split(b"=", 1) for item in raw.split(b"\0") if b"=" in item)}
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        with (RUN / "parallel_smfa_profiles.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())], env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(f"Suffix validation worker PID {process.pid}; status: {STATUS}")
    else:
        main()
