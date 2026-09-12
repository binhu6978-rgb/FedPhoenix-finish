#!/usr/bin/env python
"""Sequentially run the six original-library comparison methods.

This file is intentionally only a launcher.  Each subprocess enters the
original dispatch branch in ``main_fed.py`` so the training, aggregation,
evaluation, and result-writing logic of every baseline remains unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time


ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_METHODS = [
    "FedAvg",
    "FedProx",
    "FedMut",
    "ClusteredSampling",
    "FedGen",
    "FedPhoenix",
]
ALIASES = {
    "CluSample": "ClusteredSampling",
    "ClusteredSampling": "ClusteredSampling",
    "FedAvg": "FedAvg",
    "FedProx": "FedProx",
    "FedMut": "FedMut",
    "FedGen": "FedGen",
    "FedPhoenix": "FedPhoenix",
    "FedCRSI": "FedCRSI",
    "FedPhoenixRG": "FedPhoenixRG",
}
PARTITION_FILE = os.path.join("data", "cifar10_100_noniidCase5_beta0.3.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run six original FedPhoenix-repository baselines sequentially."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=sorted(ALIASES),
        help="methods to run in order; CluSample aliases ClusteredSampling",
    )
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=["resnet18", "vgg"],
        help="shared backbone; 'vgg' is the VGG16 protocol of result_other_method/",
    )
    parser.add_argument("--crsi-interval", type=int, default=20)
    parser.add_argument("--crsi-null-draws", type=int, default=19)
    parser.add_argument("--crsi-null-alpha", type=float, default=0.05)
    parser.add_argument("--crsi-store-backend", default="ram",
                        choices=["ram", "memmap"],
                        help="memmap keeps the ~11 GB of VGG observations on disk")
    parser.add_argument("--crsi-dry-run", type=int, default=0, choices=[0, 1])
    parser.add_argument("--rg-interval", type=int, default=20)
    parser.add_argument("--rg-strength", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop at the first failed method; default records failure and continues",
    )
    parser.add_argument(
        "--log-root",
        default="results/six_baselines_logs",
        help="launcher stdout/stderr and manifest directory",
    )
    return parser.parse_args()


def build_command(method, args):
    """Build one fixed-protocol command without changing baseline internals."""
    return [
        sys.executable,
        "-u",
        "-X",
        "utf8",
        "main_fed.py",
        "--algorithm",
        method,
        "--dataset",
        "cifar10",
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--num_users",
        "100",
        "--frac",
        "0.1",
        "--local_ep",
        "5",
        "--local_bs",
        "50",
        "--bs",
        "256",
        "--optimizer",
        "sgd",
        "--lr",
        "0.01",
        "--momentum",
        "0.5",
        "--weight_decay",
        "0",
        "--iid",
        "0",
        "--noniid_case",
        "5",
        "--data_beta",
        "0.3",
        "--FP_conv",
        "1000",
        "--FP_fc",
        "0",
        "--reset",
        "0.015625",
        "--remethod",
        "ori_normal",
        "--num_classes",
        "10",
        "--generate_data",
        "0",
        "--seed",
        str(args.seed),
        "--eval_every",
        "1",
        "--gpu",
        str(args.gpu),
        "--run_name",
        f"original_{method}_seed{args.seed}_{args.epochs}",
    ] + (method_specific_flags(method, args))


def method_specific_flags(method, args):
    """Only the proposed method receives extra flags; baselines are untouched."""
    if method == "FedPhoenixRG":
        return [
            "--rg_interval", str(args.rg_interval),
            "--rg_strength", str(args.rg_strength),
        ]
    if method != "FedCRSI":
        return []
    # store dtype is deliberately not exposed: reported runs use float32
    return [
        "--crsi_interval", str(args.crsi_interval),
        "--crsi_null", "signflip",
        "--crsi_null_draws", str(args.crsi_null_draws),
        "--crsi_null_alpha", str(args.crsi_null_alpha),
        "--crsi_store_dtype", "float32",
        "--crsi_store_backend", args.crsi_store_backend,
        "--crsi_dry_run", str(args.crsi_dry_run),
    ]


def write_manifest(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    methods = [ALIASES[name] for name in args.methods]
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.abspath(os.path.join(ROOT, args.log_root, timestamp))

    commands = [build_command(method, args) for method in methods]
    if args.dry_run:
        for method, command in zip(methods, commands):
            print(f"[{method}]")
            print(subprocess.list2cmdline(command))
        return 0

    partition_path = os.path.join(ROOT, PARTITION_FILE)
    if not os.path.isfile(partition_path):
        print(
            f"Missing client partition {PARTITION_FILE}. All methods use "
            "--generate_data 0 and must read the same fixed Dirichlet split; "
            "copy it from the FedPhoenix-new repository (do NOT regenerate).",
            file=sys.stderr,
        )
        return 2
    digest = hashlib.sha256()
    with open(partition_path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    partition_sha256 = digest.hexdigest()
    print(f"Client partition {PARTITION_FILE} sha256={partition_sha256}")
    os.makedirs(log_dir, exist_ok=False)
    manifest_path = os.path.join(log_dir, "manifest.json")
    manifest = {
        "status": "running",
        "root": ROOT,
        "python": sys.executable,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "shared_protocol": {
            "dataset": "cifar10",
            "model": args.model,
            "epochs": args.epochs,
            "num_users": 100,
            "participation": 0.1,
            "local_epochs": 5,
            "local_batch_size": 50,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "momentum": 0.5,
            "weight_decay": 0.0,
            "iid": 0,
            "noniid_case": 5,
            "dirichlet_beta": 0.3,
            "fedphoenix_conv_transition_rounds": 1000,
            "fedphoenix_reset_ratio": 0.015625,
            "fedphoenix_reset_method": "ori_normal",
            "seed": args.seed,
            "gpu": args.gpu,
            "partition_file": PARTITION_FILE,
            "partition_sha256": partition_sha256,
            "fedcrsi": {
                "interval": args.crsi_interval,
                "null": "signflip",
                "null_draws": args.crsi_null_draws,
                "null_alpha": args.crsi_null_alpha,
                "store_dtype": "float32",
                "store_backend": args.crsi_store_backend,
                "dry_run": args.crsi_dry_run,
            },
            "fedphoenix_rg": {
                "interval": args.rg_interval,
                "strength": args.rg_strength,
                "reset_ratio": 0.015625,
                "conv_transition_rounds": 1000,
                "reset_method": "ori_normal",
                "seed": args.seed,
            },
            "evaluation": (
                "original per-method evaluation loop on the complete 10000-sample "
                "CIFAR-10 server test set"
            ),
        },
        "preserves_original_method_loops": True,
        "runs": [],
    }
    write_manifest(manifest_path, manifest)

    for method, command in zip(methods, commands):
        stdout_path = os.path.join(log_dir, f"{method}.stdout.log")
        stderr_path = os.path.join(log_dir, f"{method}.stderr.log")
        record = {
            "method": method,
            "status": "running",
            "command": command,
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "stdout": stdout_path,
            "stderr": stderr_path,
        }
        manifest["runs"].append(record)
        write_manifest(manifest_path, manifest)
        print(f"Starting {method}; logs: {stdout_path}", flush=True)
        start = time.perf_counter()
        with open(stdout_path, "w", encoding="utf-8") as stdout_handle, open(
            stderr_path, "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError(f"Unable to capture stdout for {method}")
            for line in process.stdout:
                stdout_handle.write(line)
                stdout_handle.flush()
                if line.startswith(("ROUND_ACCURACY ", "PEAK_ACCURACY ")):
                    print(line, end="", flush=True)
            return_code = process.wait()
        record["exit_code"] = return_code
        record["duration_seconds"] = time.perf_counter() - start
        record["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        record["status"] = "complete" if return_code == 0 else "failed"
        write_manifest(manifest_path, manifest)
        print(f"{method}: {record['status']} (exit {return_code})", flush=True)
        if return_code != 0 and args.stop_on_error:
            manifest["status"] = "failed"
            manifest["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
            write_manifest(manifest_path, manifest)
            return return_code

    failed = [run["method"] for run in manifest["runs"] if run["status"] == "failed"]
    manifest["status"] = "complete" if not failed else "complete_with_failures"
    manifest["failed_methods"] = failed
    manifest["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    write_manifest(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
