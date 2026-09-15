"""Run the single frozen FedPhoenixRG-JSC 1000-round experiment."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
METHOD = "FedPhoenixRG-JSC"
STARTING_HEAD = "c7152f014264f3f6306d87e6705c63dbbd144280"
PARTITION = ROOT / "data" / "cifar10_100_noniidCase5_beta0.3.json"


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    if not PARTITION.is_file():
        raise FileNotFoundError(PARTITION)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS / "rg_jsc_1000" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = run_dir / f"{METHOD}.stdout.log"
    stderr_path = run_dir / f"{METHOD}.stderr.log"
    manifest_path = run_dir / "manifest.json"
    command = [
        sys.executable, "-u", "-X", "utf8", "main_fed.py",
        "--algorithm", METHOD,
        "--dataset", "cifar10", "--model", "vgg",
        "--epochs", "1000", "--num_users", "100", "--frac", "0.1",
        "--local_ep", "5", "--local_bs", "50", "--bs", "256",
        "--optimizer", "sgd", "--lr", "0.01", "--momentum", "0.5",
        "--weight_decay", "0", "--iid", "0", "--noniid_case", "5",
        "--data_beta", "0.3", "--FP_conv", "1000", "--FP_fc", "0",
        "--reset", "0.015625", "--remethod", "ori_normal",
        "--num_classes", "10", "--generate_data", "0", "--seed", "1",
        "--eval_every", "1", "--gpu", "0",
        "--run_name", "original_FedPhoenixRG-JSC_seed1_1000",
        "--rg_interval", "20", "--rg_mix", "0.75",
    ]
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    manifest = {
        "status": "running", "root": str(ROOT),
        "python": sys.executable, "starting_head": STARTING_HEAD,
        "started_at": started_at,
        "partition_file": str(PARTITION.relative_to(ROOT)),
        "partition_sha256": hashlib.sha256(PARTITION.read_bytes()).hexdigest(),
        "formal_experiment_count": 1,
        "sweep": False,
        "fixed_protocol": {
            "dataset": "cifar10", "model": "vgg", "epochs": 1000,
            "num_users": 100, "frac": 0.1, "local_ep": 5,
            "local_bs": 50, "lr": 0.01, "momentum": 0.5,
            "weight_decay": 0.0, "beta": 0.3, "seed": 1,
            "rg_mix": 0.75, "rg_interval": 20,
            "reset": 0.015625, "remethod": "ori_normal",
            "FP_conv": 1000, "FP_fc": 0, "gpu": 0,
            "aggregation": "ordinary sample-weighted absolute-state FedAvg",
            "task_sampling": "independent private-seed sampling",
        },
        "runs": [{
            "method": METHOD, "status": "running", "command": command,
            "started_at": started_at, "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }],
    }
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
            stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=stdout_handle, stderr=stderr_handle,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
    finished_at = dt.datetime.now().isoformat(timespec="seconds")
    run = manifest["runs"][0]
    run.update({
        "status": "complete" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "duration_seconds": time.perf_counter() - started,
        "finished_at": finished_at,
    })
    manifest["status"] = run["status"]
    manifest["finished_at"] = finished_at
    write_json(manifest_path, manifest)
    if completed.returncode:
        return completed.returncode
    finalizer = subprocess.run(
        [sys.executable, "-u", "-X", "utf8",
         str(RESULTS / "finalize_jsc_1000.py")], cwd=ROOT,
    )
    return finalizer.returncode


if __name__ == "__main__":
    raise SystemExit(main())
