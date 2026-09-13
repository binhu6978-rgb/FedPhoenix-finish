"""Finish the two authorized RG remapping runs once both manifests complete."""

from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
STATUS = RESULTS / "rg_remap_1000_finalize_status.json"
EXPECTED_HEAD = "2208ebb4007e9651e22eb409684970f9e2535ba5"
RUNS = {
    "FedPhoenixRG-Excess": RESULTS / "rg_remap_excess_1000",
    "FedPhoenixRG-Persistent": RESULTS / "rg_remap_persistent_1000",
}
STEM = "cifar10_vgg_{}_seed1_original_{}_seed1_1000"


def save_status(**fields):
    STATUS.write_text(
        json.dumps(
            {"updated_at": datetime.now().isoformat(), **fields},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8",
        capture_output=True, check=True,
    ).stdout.strip()


def existing_function(source, name):
    return next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def assert_existing_methods_unchanged():
    current = (ROOT / "main_fed.py").read_text(encoding="utf-8")
    previous = run_git("show", "HEAD:main_fed.py")
    for name in (
        "FedPhoenix", "FedPhoenixRG", "FedPhoenixRGLC",
        "_build_fedphoenix_tasks",
    ):
        assert ast.dump(
            existing_function(current, name), include_attributes=False
        ) == ast.dump(
            existing_function(previous, name), include_attributes=False
        ), f"existing method changed: {name}"
    scorer_path = "Algorithm/repeatability_guidance.py"
    current_scorer = ast.parse((ROOT / scorer_path).read_text(encoding="utf-8"))
    previous_scorer = ast.parse(run_git("show", f"HEAD:{scorer_path}"))
    for name in ("observe_round", "finalize_window", "reset_window"):
        def get_method(tree):
            guidance = next(
                node for node in tree.body
                if isinstance(node, ast.ClassDef)
                and node.name == "RepeatabilityGuidance"
            )
            return next(
                node for node in guidance.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
        assert ast.dump(
            get_method(current_scorer), include_attributes=False
        ) == ast.dump(
            get_method(previous_scorer), include_attributes=False
        ), f"existing scorer changed: {name}"


def completed_manifests():
    manifests = {}
    for method, directory in RUNS.items():
        candidates = list(directory.glob("*/manifest.json"))
        if len(candidates) != 1:
            return None
        manifest = json.loads(candidates[0].read_text(encoding="utf-8-sig"))
        if manifest.get("status") == "failed":
            raise RuntimeError(f"{method} failed; inspect {candidates[0]}")
        if manifest.get("status") != "complete":
            return None
        if manifest["runs"][0].get("exit_code") != 0:
            raise RuntimeError(f"{method} exited unsuccessfully")
        manifests[method] = (candidates[0], manifest)
    return manifests


def analyze(method, manifest_path):
    stem = STEM.format(method, method)
    csv_path = RESULTS / "training_metrics" / f"{stem}.csv"
    config_path = RESULTS / "training_metrics" / f"{stem}_config.json"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1000 or [int(row["round"]) for row in rows] != list(range(1, 1001)):
        raise RuntimeError(f"{method} metrics are not 1000 consecutive rounds")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in (
        ("seed", 1), ("epochs", 1000), ("rg_mix", 0.75),
        ("rg_interval", 20), ("FP_conv", 1000), ("reset", 0.015625),
    ):
        if config[key] != expected:
            raise RuntimeError(f"{method} config mismatch: {key}={config[key]}")
    peak_row = max(rows, key=lambda row: float(row["test_accuracy"]))
    stdout_path = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["runs"][0]["stdout"])
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"^RG_REMAP_SCORE .* mode=" +
                         ("excess" if method.endswith("Excess") else "persistent") +
                         r" .*sampling_entropy=([\d.]+)$", re.MULTILINE)
    diagnostic_rows = pattern.findall(stdout)
    if len(diagnostic_rows) != 650:
        # All 13 VGG convolutional layers are logged every 20 rounds,
        # including the final refresh after round 1000.
        raise RuntimeError(f"{method} has {len(diagnostic_rows)} remap log rows, expected 650")
    return {
        "peak": float(peak_row["test_accuracy"]),
        "round": int(peak_row["round"]),
        "diagnostic_rows": len(diagnostic_rows),
        "config": config_path.relative_to(ROOT).as_posix(),
        "metrics": csv_path.relative_to(ROOT).as_posix(),
        "stdout": stdout_path.relative_to(ROOT).as_posix(),
    }


def write_summary(result):
    lines = [
        "# FedPhoenixRG q-remapping: 1000-round results",
        "",
        "Peak Accuracy is the sole primary metric. Both runs use the existing",
        "CIFAR-10/VGG16 fixed beta=0.3 partition, seed 1, 1000 rounds,",
        "`rg_mix=0.75`, `rg_interval=20`, `FP_conv=1000`, reset ratio 1/64,",
        "and `ori_normal`. Existing FedPhoenix and RG references were not rerun.",
        "",
        "| Method | Peak | Peak Round | Delta vs FedPhoenix | Delta vs 83.43% |",
        "|---|---:|---:|---:|---:|",
        "| FedPhoenix | 82.61% | 997 | — | -0.82 pp |",
        "| Current-best FedPhoenixRG | **83.43%** | 984 | +0.82 pp | — |",
    ]
    for method in RUNS:
        peak = result[method]["peak"]
        lines.append(
            f"| {method} | {peak:.2f}% | {result[method]['round']} | "
            f"{peak - 82.61:+.2f} pp | {peak - 83.43:+.2f} pp |"
        )
    lines += [
        "",
        "Both experiments completed 1000 rounds with exit code 0. Each run",
        "recorded 650 per-layer remapping diagnostics alongside the original",
        "RG diagnostics. Complete logs, manifests, round metrics, and configs",
        "are in `results/rg_remap_excess_1000/`,",
        "`results/rg_remap_persistent_1000/`, and `results/training_metrics/`.",
        "",
    ]
    winners = [name for name in RUNS if result[name]["peak"] > 83.43]
    if winners:
        lines.append(
            "The method(s) exceeding 83.43% merit further analysis: "
            + ", ".join(winners) + ". No further experiment was started."
        )
    else:
        lines.append(
            "Neither remapping exceeded 83.43%. This q-remapping round stops"
            " here; no additional setting was tested."
        )
    (RESULTS / "rg_remap_1000_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    save_status(status="waiting")
    for _ in range(24 * 60):
        manifests = completed_manifests()
        if manifests is not None:
            break
        time.sleep(60)
    else:
        raise RuntimeError("timed out waiting for both 1000-round runs")
    result = {
        method: analyze(method, manifests[method][0]) for method in RUNS
    }
    if run_git("rev-parse", "HEAD") != EXPECTED_HEAD:
        raise RuntimeError("repository HEAD changed during training; manual review needed")
    if run_git("branch", "--show-current") != "main":
        raise RuntimeError("branch changed during training; manual review needed")
    if run_git("diff", "--cached", "--name-only"):
        raise RuntimeError("index has unrelated staged files; manual review needed")
    assert_existing_methods_unchanged()
    run_git("diff", "--check")
    write_summary(result)
    paths = [
        "Algorithm/repeatability_guidance.py", "main_fed.py",
        "train_five_baselines.py", "tests/test_repeatability_guidance.py",
        "tests/test_fedphoenix_rg.py", "results/finalize_rg_remap_1000.py",
        "results/rg_remap_1000_summary.md",
        "results/rg_remap_excess_1000", "results/rg_remap_persistent_1000",
    ]
    for method in RUNS:
        stem = STEM.format(method, method)
        paths.extend([
            f"results/training_metrics/{stem}.csv",
            f"results/training_metrics/{stem}_config.json",
        ])
    run_git("add", "--", *paths)
    run_git("diff", "--cached", "--check")
    run_git("commit", "-m", "Add FedPhoenixRG remapping experiments and results")
    sha = run_git("rev-parse", "HEAD")
    run_git("push", "origin", "main")
    save_status(status="pushed", commit_sha=sha, results=result)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save_status(status="failed", error=str(error))
        raise
