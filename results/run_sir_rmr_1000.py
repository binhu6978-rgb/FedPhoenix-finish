"""Launch the two frozen SIR/RMR 1000-round experiments in parallel."""

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
METHODS = ("FedPhoenixRG-SIR", "FedPhoenixRG-RMR")
RUN_ROOTS = {
    "FedPhoenixRG-SIR": RESULTS / "rg_sir_1000",
    "FedPhoenixRG-RMR": RESULTS / "rg_rmr_1000",
}
STARTING_HEAD = "82b53c0a0f3c7b1d7f30d4442dcb36901a1f4ea8"
PARTITION = ROOT / "data" / "cifar10_100_noniidCase5_beta0.3.json"


def command(method):
    return [
        sys.executable, "-u", "-X", "utf8", "main_fed.py",
        "--algorithm", method,
        "--dataset", "cifar10", "--model", "vgg",
        "--epochs", "1000", "--num_users", "100", "--frac", "0.1",
        "--local_ep", "5", "--local_bs", "50", "--bs", "256",
        "--optimizer", "sgd", "--lr", "0.01", "--momentum", "0.5",
        "--weight_decay", "0", "--iid", "0", "--noniid_case", "5",
        "--data_beta", "0.3", "--FP_conv", "1000", "--FP_fc", "0",
        "--reset", "0.015625", "--remethod", "ori_normal",
        "--num_classes", "10", "--generate_data", "0", "--seed", "1",
        "--eval_every", "1", "--gpu", "0",
        "--run_name", f"original_{method}_seed1_1000",
        "--rg_interval", "20", "--rg_mix", "0.75",
    ]


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    if not PARTITION.is_file():
        raise FileNotFoundError(PARTITION)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    partition_hash = hashlib.sha256(PARTITION.read_bytes()).hexdigest()
    jobs = []
    for method in METHODS:
        run_dir = RUN_ROOTS[method] / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
        stdout_path = run_dir / f"{method}.stdout.log"
        stderr_path = run_dir / f"{method}.stderr.log"
        manifest_path = run_dir / "manifest.json"
        cmd = command(method)
        started_at = dt.datetime.now().isoformat(timespec="seconds")
        manifest = {
            "status": "running", "root": str(ROOT),
            "python": sys.executable, "starting_head": STARTING_HEAD,
            "started_at": started_at,
            "partition_file": str(PARTITION.relative_to(ROOT)),
            "partition_sha256": partition_hash,
            "formal_experiment_count": 1, "sweep": False,
            "probability_spectrum_preserved": True,
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
                "method": method, "status": "running", "command": cmd,
                "started_at": started_at, "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }],
        }
        write_json(manifest_path, manifest)
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd, cwd=ROOT, stdout=stdout_handle, stderr=stderr_handle,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        jobs.append({
            "process": process, "stdout": stdout_handle,
            "stderr": stderr_handle, "manifest": manifest,
            "manifest_path": manifest_path, "started": time.perf_counter(),
        })

    failed = False
    for job in jobs:
        exit_code = job["process"].wait()
        job["stdout"].close()
        job["stderr"].close()
        finished_at = dt.datetime.now().isoformat(timespec="seconds")
        run = job["manifest"]["runs"][0]
        run.update({
            "status": "complete" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "duration_seconds": time.perf_counter() - job["started"],
            "finished_at": finished_at,
        })
        job["manifest"]["status"] = run["status"]
        job["manifest"]["finished_at"] = finished_at
        write_json(job["manifest_path"], job["manifest"])
        failed = failed or exit_code != 0
    if failed:
        return 1
    return subprocess.run(
        [sys.executable, "-u", "-X", "utf8",
         str(RESULTS / "finalize_sir_rmr_1000.py")], cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
