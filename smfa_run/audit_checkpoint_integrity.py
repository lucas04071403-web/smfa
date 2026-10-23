"""Check all completed formal checkpoints without waiting for evaluation."""

import json
import os
from pathlib import Path
import subprocess
import sys

import audit_final as auditor
import audit_final_names
import checkpoint_names


def main():
    import torch

    os.umask(0o077)
    torch.set_num_threads(2)
    auditor.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use CPU-only checkpoint checks")
    destination = auditor.RUN / "checkpoint_integrity_audit.json"
    auditor.require(not destination.exists(), "Refusing to replace an existing checkpoint audit")
    manifest = auditor.read_json(auditor.RUN / "manifest.json")
    auditor.require(sys.prefix == manifest["conda_prefix"], "Activate the recorded Conda environment")
    for relative, expected in manifest["source_sha256"].items():
        auditor.require(auditor.sha256(auditor.ROOT / relative) == expected,
                        f"Experiment source changed: {relative}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=auditor.REPO, text=True).strip()
    auditor.require(commit == manifest["repo_commit"], "Official repository revision changed")
    auditor.require(not subprocess.check_output(["git", "status", "--porcelain"],
                                                cwd=auditor.REPO, text=True),
                    "Official repository worktree changed")
    pipeline = auditor.read_json(auditor.RUN / "status.json")
    auditor.require("sculpt" in pipeline["completed_stages"], "Formal sculpting has not completed")
    audit_final_names.install_mapping()
    adapters, adapted_keys = auditor.audit_adapters()
    for adapter in adapters:
        epoch = auditor.read_json(auditor.RUN / f"epoch_{adapter['epoch']}_audit.json")
        previous = next(item for item in epoch["adapters"] if item["adapter"] == adapter["adapter"])
        auditor.require(adapter["sha256"] == previous["sha256"], "An audited adapter changed")
        # The original epoch-1 audit recorded weights but not configuration hashes.
        if "config_sha256" in previous:
            auditor.require(adapter["config_sha256"] == previous["config_sha256"],
                            "An audited adapter configuration changed")
    sculpted = auditor.audit_sculpted(adapted_keys)
    auditor.require(len(adapters) == 15 and sculpted["tensors"] == 765
                    and sculpted["adapted_weights"] == 274, "Checkpoint coverage differs")
    auditor.require(not torch.cuda.is_initialized(), "Checkpoint audit unexpectedly initialized CUDA")
    result = {
        "state": "passed", "checked_at": auditor.now(),
        "scope": "All formal checkpoint integrity; exact formula and full SMFA evaluation remain pending",
        "adapters": adapters, "sculpted": sculpted,
        "epoch_audit_sha256": {f"epoch_{i}_audit.json": auditor.sha256(auditor.RUN / f"epoch_{i}_audit.json")
                               for i in range(1, 6)},
        "source_sha256": manifest["source_sha256"],
        "frozen_auditor_sha256": auditor.sha256(Path(auditor.__file__)),
        "mapping_entrypoint_sha256": auditor.sha256(Path(audit_final_names.__file__)),
        "key_mapping_sha256": auditor.sha256(Path(checkpoint_names.__file__)),
        "entrypoint_sha256": auditor.sha256(Path(__file__)),
        "manifest_sha256": auditor.sha256(auditor.RUN / "manifest.json"),
        "audit_functions": ["audit_final_names.install_mapping", "audit_final.audit_adapters",
                            "audit_final.audit_sculpted"],
        "repo_commit": commit, "repo_worktree_clean": True,
        "gpu_allocations": 0, "api_calls": 0, "formal_inputs_modified": False,
    }
    auditor.write_json(destination, result)
    print(json.dumps({"state": "passed", "adapters": len(adapters),
                      "sculpted_tensors": sculpted["tensors"],
                      "adapted_weights": sculpted["adapted_weights"],
                      "changed_weights": sculpted["changed_weights"], "evidence": str(destination)}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
