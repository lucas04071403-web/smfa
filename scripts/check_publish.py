"""Check staged Git blobs before publishing code and aggregate experiment evidence."""

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1024 * 1024
DENIED_DIRS = {"data", "datasets", "model", "models", "checkpoints", "eval_result", "outputs",
               "logs", "runs", "wandb", ".profiles", "__pycache__", ".venv", ".conda", ".git"}
DENIED_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".npy", ".npz",
                   ".parquet", ".arrow", ".jsonl", ".csv", ".tsv", ".log", ".pid", ".lock", ".swap",
                   ".pdf", ".zip", ".tar", ".gz", ".7z", ".png", ".jpg", ".jpeg", ".webp", ".mp4"}
SECRET_PATTERNS = [
    rb"\b(?:sk-[A-Za-z0-9_-]{18,}|gh[pousr]_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{16,}|hf_[A-Za-z0-9]{20,})\b",
    rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----",
    rb"https?://[^\s/:]+:[^\s/@]+@",
]
ALLOWED_FIGURES = {'wp1_a_stage/A_tradeoff.png',
                   'wp1_supplement/figures/c1_enrichment.png',
                   'wp1_supplement/figures/supplement_tradeoffs.png'}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    records = [x for x in git("ls-files", "--stage", "-z").split(b"\0") if x]
    errors = []
    total = largest = 0
    largest_name = ""
    for entry in records:
        info, raw_name = entry.split(b"\t", 1)
        mode, oid, stage = info.decode().split()
        name = raw_name.decode()
        path = Path(name)
        if mode not in {"100644", "100755"} or stage != "0":
            errors.append(f"Unsupported index entry (symlink/submodule/conflict): {name}")
            continue
        if DENIED_DIRS.intersection(path.parts) or (path.suffix.lower() in DENIED_SUFFIXES and name not in ALLOWED_FIGURES):
            errors.append(f"Excluded dataset/model/runtime path: {name}")
        if path.name.startswith((".env", "id_rsa", "id_ed25519")) or path.suffix in {".pem", ".key"}:
            errors.append(f"Credential filename: {name}")
        blob = git("cat-file", "blob", oid)
        total += len(blob)
        if len(blob) > largest:
            largest, largest_name = len(blob), name
        if len(blob) > MAX_BYTES:
            errors.append(f"File exceeds {MAX_BYTES} bytes: {name}")
        if any(re.search(pattern, blob) for pattern in SECRET_PATTERNS):
            errors.append(f"Potential credential (value withheld): {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps({"state": "passed", "files": len(records), "total_bytes": total,
                      "largest_file": largest_name, "largest_bytes": largest,
                      "maximum_allowed_file_bytes": MAX_BYTES}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
