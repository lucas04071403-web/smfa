"""Download and verify the four checkpoints listed by S-MLLMUn."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

# Keep the HTTP proxy while avoiding an unused SOCKS transport in httpx.
if os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("all_proxy", None)
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent / "S-MLLMUn" / "model"
CHUNK_SIZE = 8 * 1024 * 1024
TOTAL_BYTES = 81_400_554_005
progress_lock = threading.Lock()
completed_bytes = 0
last_report = 0.0
MODELS = {
    "llava-hf/llava-onevision-qwen2-7b-si-hf": (
        "base_model/llava-onevision-qwen2-7b-si-hf"
    ),
    "Qwen/Qwen2.5-VL-7B-Instruct": "base_model/Qwen2.5-VL-7B-Instruct",
    "ZhenZeng/S-MLLMUn-llava-onevision-qwen2-7b-original": (
        "original_model/llava-onevision-qwen2-7b-original"
    ),
    "ZhenZeng/S-MLLMUn-Qwen2.5-VL-7B-original": (
        "original_model/Qwen2.5-VL-7B-original"
    ),
}


def report_progress(size):
    global completed_bytes, last_report
    with progress_lock:
        completed_bytes += size
        now = time.monotonic()
        if now - last_report >= 30:
            status = {
                'state': 'downloading',
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'completed_bytes': completed_bytes,
                'total_bytes': TOTAL_BYTES,
                'pid': os.getpid(),
            }
            status_path = ROOT / 'DOWNLOAD_STATUS.json'
            status_partial = status_path.with_suffix('.json.part')
            status_partial.write_text(json.dumps(status, indent=2) + '\n')
            status_partial.replace(status_path)
            print(
                f"PROGRESS {completed_bytes / 1e9:.2f}/{TOTAL_BYTES / 1e9:.2f} GB "
                f"({completed_bytes / TOTAL_BYTES:.1%})",
                flush=True,
            )
            last_report = now


def download_chunk(url, path, start, end, full_size):
    attempt = 0
    while True:
        try:
            return download_chunk_once(url, path, start, end, full_size)
        except RuntimeError as error:
            attempt += 1
            delay = min(10 * attempt, 60)
            print(
                f"RETRY {path.parent.name}/{path.name} attempt={attempt} "
                f"in={delay}s reason={str(error)[-240:]}",
                flush=True,
            )
            report_progress(0)
            time.sleep(delay)


def download_chunk_once(url, path, start, end, full_size):
    size = end - start + 1
    if path.is_file() and path.stat().st_size == size:
        report_progress(size)
        return path
    partial = path.with_suffix(".part")
    headers = path.with_suffix(".headers")
    command = [
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--connect-timeout", "20", "--max-time", "90",
        "--speed-limit", "1024", "--speed-time", "30",
        "--retry", "2", "--retry-all-errors", "--retry-delay", "2",
        "--max-filesize", str(size), "--output", str(partial),
        "--dump-header", str(headers),
        "--write-out", "%{http_code}",
    ]
    if size < full_size:
        command.extend(["--range", f"{start}-{end}"])
    range_url = url + f'&range_start={start}&range_end={end}'
    result = subprocess.run(command + [range_url], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Chunk download failed: {path}: {result.stderr[-500:]}")
    expected_status = "206" if size < full_size else "200"
    if not result.stdout.endswith(expected_status) or partial.stat().st_size != size:
        raise RuntimeError(f"Invalid range response for {path}: {result.stdout}")
    if size < full_size:
        ranges = re.findall(r'^content-range:\s*bytes (\d+)-(\d+)/(\d+)', headers.read_text(), re.I | re.M)
        if not ranges or tuple(map(int, ranges[-1])) != (start, end, full_size):
            raise RuntimeError(f'Wrong content range for {path}: {ranges[-1:]}')
    partial.replace(path)
    headers.unlink(missing_ok=True)
    report_progress(size)
    return path


def download_asset(repo_id, revision, entry, directory, pool):
    path = directory / entry.rfilename
    path.parent.mkdir(parents=True, exist_ok=True)
    aria_control = path.with_name(path.name + ".aria2")
    if not aria_control.exists() and path.is_file() and path.stat().st_size == entry.size:
        report_progress(entry.size)
        return
    cache = directory / ".cache" / "curl_8MiB" / revision / entry.rfilename
    cache.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://huggingface.co/{repo_id}/resolve/{revision}/"
        f"{quote(entry.rfilename, safe='/')}?download=true"
    )
    pieces = []
    futures = []
    for start in range(0, entry.size, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, entry.size) - 1
        piece = cache / f"{start:012d}.chunk"
        pieces.append(piece)
        futures.append(pool.submit(download_chunk, url, piece, start, end, entry.size))
    for future in as_completed(futures):
        future.result()
    partial = path.with_name(path.name + ".part")
    with partial.open("wb") as output:
        for piece in pieces:
            with piece.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    partial.replace(path)
    aria_control.unlink(missing_ok=True)
    print(f"DOWNLOADED {repo_id}/{entry.rfilename}", flush=True)


def download_and_verify(repo_id, relative_path, pool):
    info = HfApi(token=False).model_info(repo_id, files_metadata=True)
    directory = ROOT / relative_path
    print(
        f"START {repo_id} revision={info.sha} "
        f"bytes={sum(f.size for f in info.siblings)}",
        flush=True,
    )
    directory.mkdir(parents=True, exist_ok=True)
    for entry in info.siblings:
        download_asset(repo_id, info.sha, entry, directory, pool)
    files = []
    for entry in info.siblings:
        path = directory / entry.rfilename
        if path.stat().st_size != entry.size:
            raise RuntimeError(f"Size mismatch: {path}")
        if entry.lfs:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            expected = entry.lfs.sha256
            algorithm = "sha256"
        else:
            content = path.read_bytes()
            digest = hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content
            ).hexdigest()
            expected = entry.blob_id
            algorithm = "git_blob_sha1"
        if digest != expected:
            # A restart must redownload a corrupt file instead of skipping it
            # merely because its length happens to be correct.
            path.unlink()
            cache = directory / ".cache" / "curl_8MiB" / info.sha / entry.rfilename
            if cache.exists():
                shutil.rmtree(cache)
            raise RuntimeError(f"Hash mismatch: {path}")
        cache = directory / ".cache" / "curl_8MiB" / info.sha / entry.rfilename
        if cache.exists():
            shutil.rmtree(cache)
        files.append(
            {
                "path": entry.rfilename,
                "bytes": entry.size,
                "algorithm": algorithm,
                "hash": digest,
            }
        )
        print(f"VERIFIED {relative_path}/{entry.rfilename}", flush=True)

    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    for shard in shards:
        if not (directory / shard).is_file():
            raise RuntimeError(f"Missing indexed shard: {directory / shard}")
    print(f"COMPLETE {repo_id}: {len(shards)} weight shards verified", flush=True)
    return {
        "repo_id": repo_id,
        "revision": info.sha,
        "directory": str(directory),
        "weight_shards": len(shards),
        "files": files,
    }


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=12) as chunks, ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(download_and_verify, repo_id, directory, chunks)
            for repo_id, directory in MODELS.items()
        ]
        for future in as_completed(futures):
            records.append(future.result())
            manifest = {
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "models": sorted(records, key=lambda record: record["repo_id"]),
            }
            manifest_path = ROOT / "DOWNLOAD_MANIFEST.json"
            partial = manifest_path.with_suffix(".json.part")
            partial.write_text(json.dumps(manifest, indent=2) + "\n")
            partial.replace(manifest_path)
    print(f"ALL VERIFIED: {len(records)} models. Manifest: {manifest_path}", flush=True)
    (ROOT / 'DOWNLOAD_STATUS.json').write_text(json.dumps({
        'state': 'complete', 'updated_at': datetime.now(timezone.utc).isoformat(),
        'completed_bytes': TOTAL_BYTES, 'total_bytes': TOTAL_BYTES,
        'verified_models': len(records),
    }, indent=2) + '\n')


def run_worker():
    try:
        main()
    except Exception as error:
        (ROOT / 'DOWNLOAD_STATUS.json').write_text(json.dumps({
            'state': 'failed', 'updated_at': datetime.now(timezone.utc).isoformat(),
            'error': str(error), 'completed_bytes': completed_bytes,
            'total_bytes': TOTAL_BYTES,
        }, indent=2) + '\n')
        raise


if __name__ == "__main__":
    if "--supervise" in sys.argv:
        attempt = 0
        while True:
            print(f"SUPERVISOR starting worker at {datetime.now(timezone.utc).isoformat()}", flush=True)
            result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve())])
            if result.returncode == 0:
                break
            attempt += 1
            delay = min(attempt * 15, 60)
            print(f"SUPERVISOR worker exited={result.returncode}; restarting in {delay}s", flush=True)
            time.sleep(delay)
    else:
        run_worker()
