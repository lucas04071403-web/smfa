"""Run the frozen final audit with verified runtime-to-checkpoint LoRA names."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import audit_final as auditor
import checkpoint_names

RUN = auditor.RUN


def install_mapping():
    checkpoint_names.verify_installed_mapping()
    original_check = auditor.audit_sculpted
    original_final = auditor.final_audit

    def sculpted_check(adapted_keys):
        renamed = {key: checkpoint_names.runtime_to_checkpoint(key) for key in adapted_keys}
        auditor.require(len(set(renamed.values())) == len(renamed), "Adapted layer name mapping is not bijective")
        result = original_check(set(renamed.values()))
        result.update(runtime_to_checkpoint_adapted_keys=renamed,
                      checkpoint_conversion_mapping=checkpoint_names.CONVERSION)
        return result

    def final_check():
        result = original_final()
        result.update(final_audit_entrypoint="smfa_run/audit_final_names.py",
                      final_audit_entrypoint_sha256=auditor.sha256(Path(__file__)),
                      frozen_final_auditor_sha256=auditor.sha256(Path(auditor.__file__)),
                      key_mapping_sha256=auditor.sha256(Path(checkpoint_names.__file__)))
        return result

    auditor.audit_sculpted = sculpted_check
    auditor.final_audit = final_check


def main():
    os.umask(0o077)
    with (RUN / "final_audit_names.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen_hashes = {str(Path(module.__file__)): auditor.sha256(Path(module.__file__))
                         for module in (auditor, checkpoint_names)}
        own_hash = auditor.sha256(Path(__file__))
        state = {"pid": os.getpid(), "state": "running", "stage": "waiting_for_original_final_auditor",
                 "started_at": auditor.now(), "launcher_sha256": own_hash, "source_sha256": frozen_hashes}
        status_path = RUN / "final_audit_names_status.json"
        auditor.write_json(status_path, state)
        try:
            while True:
                pipeline = auditor.read_json(RUN / "status.json")
                auditor.require(pipeline["state"] in ("running", "complete"), "Formal pipeline stopped")
                auditor.require(pipeline["state"] == "complete" or auditor.live(pipeline["pid"]),
                                "Formal pipeline controller exited")
                original = auditor.read_json(RUN / "final_audit_status.json")
                if pipeline["state"] == "complete" and not auditor.live(original["pid"]):
                    if original["state"] == "complete":
                        state.update(state="complete", stage="original_auditor_passed",
                                     completed_at=auditor.now(), original_audit_sha256=original.get("audit_sha256"))
                        auditor.write_json(status_path, state)
                        return
                    auditor.require(original["state"] == "failed"
                                    and original.get("error") == "Sculpted checkpoint is missing adapted weights",
                                    "Original final audit failed for a different reason; inspect it")
                    break
                time.sleep(30)
            auditor.require(auditor.sha256(Path(__file__)) == own_hash, "Audit entrypoint changed after launch")
            for path, expected in frozen_hashes.items():
                auditor.require(auditor.sha256(Path(path)) == expected, "Frozen audit or key mapping changed")
            stamp = auditor.now().replace(":", "").replace("+", "_")
            backup = RUN / f"final_audit_name_mapping_recovery_{stamp}"
            backup.mkdir()
            for name in ("final_audit_status.json", "final_audit.log"):
                shutil.copy2(RUN / name, backup / name)
            auditor.require(not (RUN / "final_audit.json").exists(), "Final audit output already exists")
            auditor.require(not (auditor.ROOT / "SMFA_FINAL_AUDIT.md").exists(), "Final audit report already exists")
            state.update(stage="running_final_audit_with_saved_names", backup=str(backup), checked_at=auditor.now())
            auditor.write_json(status_path, state)
            install_mapping()
            auditor.main(False)
            final = auditor.read_json(RUN / "final_audit_status.json")
            auditor.require(final["state"] == "complete", "Mapped-name final audit did not complete")
            state.update(state="complete", stage="complete", completed_at=auditor.now(),
                         audit_sha256=final["audit_sha256"])
            auditor.write_json(status_path, state)
        except BaseException as error:
            state.update(state="failed", error=str(error), failed_at=auditor.now())
            auditor.write_json(status_path, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        os.umask(0o077)
        with (RUN / "final_audit_names.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())],
                                       env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Launched saved-name final audit watcher PID {process.pid}")
    else:
        main()
