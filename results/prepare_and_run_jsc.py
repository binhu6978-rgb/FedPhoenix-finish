"""Validate the 20-round JSC prefix, wait for a free GPU, then run JSC."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PILOT = RESULTS / "jsc_first20_validation"
PILOT_STEM = "cifar10_vgg_FedPhoenixRG-JSC_seed1_jsc_first20_validation"
PILOT_METRICS = PILOT / f"{PILOT_STEM}.csv"
REFERENCE = RESULTS / "training_metrics" / (
    "cifar10_vgg_FedPhoenixRG_seed1_paired1200_rg_mix075.csv"
)
STATUS = RESULTS / "jsc_pipeline_status.json"
STARTING_HEAD = "c7152f014264f3f6306d87e6705c63dbbd144280"


def save_status(status, **fields):
    STATUS.write_text(
        json.dumps(
            {"updated_at": datetime.now().isoformat(), "status": status, **fields},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8",
    )


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def gpu_memory_mib():
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True, encoding="utf-8", capture_output=True, check=True,
    )
    return int(completed.stdout.strip().splitlines()[0])


def main():
    save_status("waiting_for_first20")
    for _ in range(24 * 60 * 4):
        if PILOT_METRICS.is_file():
            break
        time.sleep(15)
    else:
        raise RuntimeError("timed out waiting for the 20-round validation")

    pilot = load_csv(PILOT_METRICS)
    reference = load_csv(REFERENCE)[:20]
    if len(pilot) != 20:
        raise RuntimeError(f"validation produced {len(pilot)} rounds")
    comparisons = {
        "rounds": 20,
        "selected_clients_match": [row["selected_clients"] for row in pilot]
        == [row["selected_clients"] for row in reference],
        "task_seeds_match": [row["task_seeds"] for row in pilot]
        == [row["task_seeds"] for row in reference],
        "round_accuracies_match": [row["test_accuracy"] for row in pilot]
        == [row["test_accuracy"] for row in reference],
        "maximum_accuracy_difference": max(
            abs(float(left["test_accuracy"]) - float(right["test_accuracy"]))
            for left, right in zip(pilot, reference)
        ),
        "reference_metrics": str(REFERENCE.relative_to(ROOT)),
    }
    if not all(
        comparisons[key] for key in (
            "selected_clients_match", "task_seeds_match", "round_accuracies_match"
        )
    ):
        raise RuntimeError(f"first-20 validation mismatch: {comparisons}")
    comparison_path = PILOT / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for path in PILOT.iterdir():
        if path != comparison_path and path.is_file():
            path.unlink()

    save_status("waiting_for_gpu", first20=comparisons)
    consecutive_free_checks = 0
    for _ in range(24 * 60 * 2):
        memory = gpu_memory_mib()
        if memory < 5000:
            consecutive_free_checks += 1
        else:
            consecutive_free_checks = 0
        if consecutive_free_checks >= 2:
            break
        time.sleep(30)
    else:
        raise RuntimeError("GPU did not become available within 24 hours")

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    if head != STARTING_HEAD:
        raise RuntimeError(f"HEAD changed before formal launch: {head}")
    if (RESULTS / "rg_jsc_1000").exists():
        raise RuntimeError("formal JSC output directory already exists")
    save_status("formal_running", first20=comparisons)
    completed = subprocess.run(
        [sys.executable, "-u", "-X", "utf8", str(RESULTS / "run_jsc_1000.py")],
        cwd=ROOT,
    )
    if completed.returncode:
        raise RuntimeError(f"formal JSC pipeline exited {completed.returncode}")
    save_status("complete", first20=comparisons)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save_status("failed", error=repr(error))
        raise
