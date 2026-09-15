"""Launch the two fixed RGE/RCC 1000-round experiments in parallel."""

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
PYTHON = Path(sys.executable)
METHODS = ("FedPhoenixRG-RGE", "FedPhoenixRG-RCC")
RUN_ROOTS = {
    "FedPhoenixRG-RGE": RESULTS / "rg_rge_1000",
    "FedPhoenixRG-RCC": RESULTS / "rg_rcc_1000",
}
PARTITION = ROOT / "data" / "cifar10_100_noniidCase5_beta0.3.json"


def command(method):
    return [
        str(PYTHON), "-u", "-X", "utf8", "main_fed.py",
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


def write_manifest(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    if not PARTITION.is_file():
        raise FileNotFoundError(PARTITION)
    partition_hash = hashlib.sha256(PARTITION.read_bytes()).hexdigest()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    processes = []
    for method in METHODS:
        run_dir = RUN_ROOTS[method] / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
        stdout_path = run_dir / f"{method}.stdout.log"
        stderr_path = run_dir / f"{method}.stderr.log"
        manifest_path = run_dir / "manifest.json"
        cmd = command(method)
        started = dt.datetime.now().isoformat(timespec="seconds")
        manifest = {
            "status": "running",
            "root": str(ROOT),
            "python": str(PYTHON),
            "starting_head": "4c659969d28f35f302d07452cb315094fccbbcf5",
            "started_at": started,
            "partition_file": str(PARTITION.relative_to(ROOT)),
            "partition_sha256": partition_hash,
            "fixed_protocol": {
                "dataset": "cifar10", "model": "vgg", "epochs": 1000,
                "num_users": 100, "frac": 0.1, "local_ep": 5,
                "local_bs": 50, "lr": 0.01, "momentum": 0.5,
                "weight_decay": 0.0, "beta": 0.3, "seed": 1,
                "rg_mix": 0.75, "rg_interval": 20,
                "reset": 0.015625, "remethod": "ori_normal",
                "FP_conv": 1000, "FP_fc": 0, "gpu": 0,
            },
            "runs": [{
                "method": method, "status": "running", "command": cmd,
                "started_at": started, "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }],
        }
        write_manifest(manifest_path, manifest)
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd, cwd=ROOT, stdout=stdout_handle, stderr=stderr_handle,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        processes.append({
            "method": method, "process": process,
            "stdout_handle": stdout_handle, "stderr_handle": stderr_handle,
            "manifest_path": manifest_path, "manifest": manifest,
            "start_time": time.perf_counter(),
        })

    any_failed = False
    for item in processes:
        exit_code = item["process"].wait()
        item["stdout_handle"].close()
        item["stderr_handle"].close()
        finished = dt.datetime.now().isoformat(timespec="seconds")
        run = item["manifest"]["runs"][0]
        run.update({
            "status": "complete" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "duration_seconds": time.perf_counter() - item["start_time"],
            "finished_at": finished,
        })
        item["manifest"]["status"] = run["status"]
        item["manifest"]["finished_at"] = finished
        write_manifest(item["manifest_path"], item["manifest"])
        any_failed = any_failed or exit_code != 0

    if any_failed:
        return 1
    completed = subprocess.run(
        [str(PYTHON), "-u", "-X", "utf8",
         str(RESULTS / "finalize_rge_rcc_1000.py")],
        cwd=ROOT,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
