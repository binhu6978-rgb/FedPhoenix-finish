"""Validate, summarize, commit, and push the fixed RGE/RCC experiments."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import statistics
import subprocess
from datetime import datetime


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
STATUS = RESULTS / "rge_rcc_1000_finalize_status.json"
STARTING_HEAD = "4c659969d28f35f302d07452cb315094fccbbcf5"
METHODS = ("FedPhoenixRG-RGE", "FedPhoenixRG-RCC")
RUN_ROOTS = {
    "FedPhoenixRG-RGE": RESULTS / "rg_rge_1000",
    "FedPhoenixRG-RCC": RESULTS / "rg_rcc_1000",
}


def save_status(**fields):
    STATUS.write_text(
        json.dumps(
            {"updated_at": datetime.now().isoformat(), **fields},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8",
        capture_output=True, check=True,
    ).stdout.strip()


def function(source, name):
    return next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def class_method(source, class_name, method_name):
    cls = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def assert_frozen_behavior():
    current_main = (ROOT / "main_fed.py").read_text(encoding="utf-8")
    baseline_main = git("show", f"{STARTING_HEAD}:main_fed.py")
    for name in ("FedPhoenix", "FedPhoenixRG", "_build_fedphoenix_tasks"):
        assert ast.dump(function(current_main, name), include_attributes=False) == ast.dump(
            function(baseline_main, name), include_attributes=False
        ), f"frozen function changed: {name}"
    path = "Algorithm/repeatability_guidance.py"
    current_q = (ROOT / path).read_text(encoding="utf-8")
    baseline_q = git("show", f"{STARTING_HEAD}:{path}")
    for name in ("observe_round", "finalize_window", "reset_window"):
        assert ast.dump(
            class_method(current_q, "RepeatabilityGuidance", name),
            include_attributes=False,
        ) == ast.dump(
            class_method(baseline_q, "RepeatabilityGuidance", name),
            include_attributes=False,
        ), f"q estimator changed: {name}"


def manifest_for(method):
    candidates = sorted(RUN_ROOTS[method].glob("*/manifest.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {method} manifest, found {len(candidates)}")
    path = candidates[0]
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    run = payload["runs"][0]
    if payload.get("status") != "complete" or run.get("exit_code") != 0:
        raise RuntimeError(f"{method} did not complete successfully: {path}")
    if len(payload["runs"]) != 1 or run["method"] != method:
        raise RuntimeError(f"unexpected extra/combined run in {path}")
    return path, payload


def load_run(method, manifest_path):
    stem = f"cifar10_vgg_{method}_seed1_original_{method}_seed1_1000"
    metrics_path = RESULTS / "training_metrics" / f"{stem}.csv"
    config_path = RESULTS / "training_metrics" / f"{stem}_config.json"
    diag_suffix = "rge_diagnostics" if method.endswith("RGE") else "rcc_diagnostics"
    diagnostic_path = RESULTS / "training_metrics" / f"{stem}_{diag_suffix}.jsonl"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1000 or [int(row["round"]) for row in rows] != list(range(1, 1001)):
        raise RuntimeError(f"{method} metrics are not 1000 consecutive rounds")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "algorithm": method, "epochs": 1000, "seed": 1,
        "dataset": "cifar10", "model": "vgg", "num_users": 100,
        "frac": 0.1, "local_ep": 5, "local_bs": 50,
        "lr": 0.01, "momentum": 0.5, "weight_decay": 0.0,
        "rg_mix": 0.75, "rg_interval": 20, "reset": 0.015625,
        "remethod": "ori_normal", "FP_conv": 1000, "FP_fc": 0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"{method} config mismatch: {key}={config.get(key)}")
    with diagnostic_path.open(encoding="utf-8") as handle:
        diagnostics = [json.loads(line) for line in handle if line.strip()]
    accuracies = [float(row["test_accuracy"]) for row in rows]
    peak_index = max(range(1000), key=accuracies.__getitem__)
    neighborhood = accuracies[max(0, peak_index - 5):min(1000, peak_index + 6)]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    return {
        "peak": accuracies[peak_index], "peak_round": peak_index + 1,
        "round_1000": accuracies[-1],
        "neighborhood_start": max(1, peak_index - 4),
        "neighborhood_end": min(1000, peak_index + 6),
        "neighborhood_mean": statistics.mean(neighborhood),
        "neighborhood_second_best": sorted(neighborhood, reverse=True)[1],
        "neighborhood_above_reference": sum(value > 83.43 for value in neighborhood),
        "selected_clients": [row["selected_clients"] for row in rows],
        "task_seeds": [row["task_seeds"] for row in rows],
        "diagnostics": diagnostics,
        "metrics_path": metrics_path.relative_to(ROOT).as_posix(),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "diagnostic_path": diagnostic_path.relative_to(ROOT).as_posix(),
        "stdout_path": Path(manifest["runs"][0]["stdout"]).relative_to(ROOT).as_posix(),
        "stderr_path": Path(manifest["runs"][0]["stderr"]).relative_to(ROOT).as_posix(),
    }


def verify_sequences(runs):
    reference_path = RESULTS / "training_metrics" / (
        "cifar10_vgg_FedPhoenixRG_seed1_paired1200_rg_mix075.csv"
    )
    with reference_path.open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))[:1000]
    clients = [row["selected_clients"] for row in reference]
    seeds = [row["task_seeds"] for row in reference]
    for method, run in runs.items():
        if run["selected_clients"] != clients:
            raise RuntimeError(f"{method} changed client sampling")
        if run["task_seeds"] != seeds:
            raise RuntimeError(f"{method} changed task seed sequence")


def rge_summary(rows):
    gates = [row for row in rows if row.get("event") == "rge_gate"]
    extended = [row for row in rows if row.get("event") == "rge_extended_round"]
    triggered = [row for row in gates if row["gate_triggered"]]
    by_layer = {}
    for gate in gates:
        layer = gate["layer_name"]
        active_rows = [row for row in extended if row["layer_name"] == layer]
        by_layer[layer] = {
            "original_stop_round": gate["original_stop_round"],
            "latest_q_at_stop": gate["latest_q_at_stop"],
            "previous_active_mean_q": gate["previous_active_mean_q"],
            "retention_ratio": gate["retention_ratio"],
            "gate_triggered": gate["gate_triggered"],
            "extended_rounds_logged": len(active_rows),
            "extended_mean_q": (
                statistics.mean(row["mean_q"] for row in active_rows)
                if active_rows else None
            ),
            "extended_mean_entropy": (
                statistics.mean(row["sampling_entropy"] for row in active_rows)
                if active_rows else None
            ),
            "extended_mean_accuracy": (
                statistics.mean(row["test_accuracy"] for row in active_rows)
                if active_rows else None
            ),
        }
    return {"gate_count": len(gates), "triggered_count": len(triggered), "layers": by_layer}


def weighted(rows, key):
    total = sum(row["total_reset_slots"] for row in rows)
    return sum(row[key] * row["total_reset_slots"] for row in rows) / total


def rcc_summary(rows):
    actual = [
        row for row in rows
        if row.get("event") == "rcc_actual" and row["round"] >= 21
    ]
    counter = [
        row for row in rows
        if row.get("event") == "rcc_independent_counterfactual"
        and row["round"] >= 21
    ]
    for group in (actual, counter):
        if not group:
            raise RuntimeError("missing RCC coverage diagnostics")
    total_actual = sum(row["total_reset_slots"] for row in actual)
    total_counter = sum(row["total_reset_slots"] for row in counter)
    return {
        "guided_layer_rounds": len(actual),
        "actual_total_slots": total_actual,
        "actual_duplicate_slots": sum(row["duplicate_slots"] for row in actual),
        "actual_fraction_unique": 1.0 - sum(row["duplicate_slots"] for row in actual) / total_actual,
        "counterfactual_total_slots": total_counter,
        "counterfactual_duplicate_slots": sum(row["duplicate_slots"] for row in counter),
        "counterfactual_fraction_unique": 1.0 - sum(row["duplicate_slots"] for row in counter) / total_counter,
        "actual_mean_selected_q": weighted(actual, "mean_selected_q"),
        "counterfactual_mean_selected_q": weighted(counter, "mean_selected_q"),
        "actual_mean_selected_probability": weighted(actual, "mean_selected_probability"),
        "counterfactual_mean_selected_probability": weighted(counter, "mean_selected_probability"),
        "actual_max_multiplicity": max(row["max_multiplicity"] for row in actual),
        "counterfactual_max_multiplicity": max(row["max_multiplicity"] for row in counter),
    }


def result_row(method, run):
    return (
        f"| {method} | {run['peak']:.2f}% | {run['peak_round']} | "
        f"{run['peak'] - 82.61:+.2f} pp | {run['peak'] - 83.43:+.2f} pp | "
        f"{run['round_1000']:.2f}% | {run['neighborhood_start']}–"
        f"{run['neighborhood_end']} | {run['neighborhood_mean']:.2f}% | "
        f"{run['neighborhood_second_best']:.2f}% |"
    )


def write_summary(runs, rge, rcc):
    lines = [
        "# FedPhoenixRG RGE/RCC experiments", "",
        f"Starting HEAD: `{STARTING_HEAD}` on `main`.", "",
        "## Results", "",
        "| Method | Peak | Peak round | vs FedPhoenix | vs FedPhoenixRG | Round 1000 | Peak ±5 rounds | Neighborhood mean | Neighborhood second-best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| FedPhoenix | 82.61% | 997 | — | -0.82 pp | 80.59% | — | — | — |",
        "| Current-best FedPhoenixRG | **83.43%** | 984 | +0.82 pp | — | 81.27% | 979–989 | 81.84% | 83.39% |",
        result_row("FedPhoenixRG-RGE", runs["FedPhoenixRG-RGE"]),
        result_row("FedPhoenixRG-RCC", runs["FedPhoenixRG-RCC"]), "",
        "## RGE mechanism diagnostics", "",
        f"The gate triggered for {rge['triggered_count']} of {rge['gate_count']} layer boundaries. Details:", "",
        "| Layer | Original stop round | Latest q | Earlier mean q | Retention | Triggered | Logged extension rounds | Extension mean q | Extension mean entropy | Extension mean accuracy |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for layer, row in rge["layers"].items():
        def fmt(value):
            return "—" if value is None else f"{value:.6f}"
        lines.append(
            f"| `{layer}` | {row['original_stop_round']} | {row['latest_q_at_stop']:.6f} | "
            f"{row['previous_active_mean_q']:.6f} | {row['retention_ratio']:.6f} | "
            f"{'yes' if row['gate_triggered'] else 'no'} | {row['extended_rounds_logged']} | "
            f"{fmt(row['extended_mean_q'])} | {fmt(row['extended_mean_entropy'])} | "
            f"{fmt(row['extended_mean_accuracy'])} |"
        )
    lines += [
        "", "## RCC mechanism diagnostics", "",
        f"Across {rcc['guided_layer_rounds']} guided layer-rounds, coordinated sampling used "
        f"{rcc['actual_total_slots']} reset slots and produced "
        f"{rcc['actual_duplicate_slots']} duplicate slots "
        f"({rcc['actual_fraction_unique']:.2%} unique). The private independent "
        f"counterfactual produced {rcc['counterfactual_duplicate_slots']} duplicate slots "
        f"({rcc['counterfactual_fraction_unique']:.2%} unique).", "",
        f"Mean selected q was {rcc['actual_mean_selected_q']:.6f} for RCC and "
        f"{rcc['counterfactual_mean_selected_q']:.6f} for the independent counterfactual. "
        f"Mean selected probability was {rcc['actual_mean_selected_probability']:.6f} "
        f"versus {rcc['counterfactual_mean_selected_probability']:.6f}; maximum "
        f"multiplicity was {rcc['actual_max_multiplicity']} versus "
        f"{rcc['counterfactual_max_multiplicity']}.", "",
        "## Interpretation", "",
    ]
    both_fail = all(run["peak"] < 83.70 for run in runs.values())
    if both_fail:
        lines.append(
            "Neither independent mechanism reached the approximately 83.7% threshold "
            "for a clear single-seed improvement with neighborhood support. Under the "
            "fixed framework, current FedPhoenixRG is likely near a local performance "
            "ceiling. No further seed=1 method search was launched."
        )
    else:
        lines.append(
            "At least one independent mechanism reached approximately 83.7%; its peak "
            "neighborhood above determines whether it warrants further analysis. No "
            "follow-up was launched automatically."
        )
    lines += [
        "", "Original `FedPhoenix` and `FedPhoenixRG` function bodies and the q "
        "estimator remained unchanged. Exactly two independent runs were executed; "
        "there was no combined or third variant.", "",
        "Focused validation: 27 tests passed before formal launch.",
    ]
    (RESULTS / "rge_rcc_1000_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    save_status(status="validating")
    if git("rev-parse", "HEAD") != STARTING_HEAD:
        raise RuntimeError("HEAD changed during training; manual review required")
    if git("branch", "--show-current") != "main":
        raise RuntimeError("branch changed during training")
    manifests = {method: manifest_for(method) for method in METHODS}
    runs = {
        method: load_run(method, manifests[method][0]) for method in METHODS
    }
    verify_sequences(runs)
    assert_frozen_behavior()
    rge = rge_summary(runs["FedPhoenixRG-RGE"]["diagnostics"])
    rcc = rcc_summary(runs["FedPhoenixRG-RCC"]["diagnostics"])
    git("diff", "--check")
    write_summary(runs, rge, rcc)
    paths = [
        "Algorithm/Phoenix_util.py", "Algorithm/timing_coverage_variants.py",
        "main_fed.py", "train_five_baselines.py",
        "tests/test_timing_coverage_variants.py",
        "results/rge_rcc_reference_analysis.md",
        "results/run_rge_rcc_1000.py", "results/finalize_rge_rcc_1000.py",
        "results/rge_rcc_1000_summary.md", "results/rg_rge_1000",
        "results/rg_rcc_1000",
    ]
    for method in METHODS:
        stem = f"cifar10_vgg_{method}_seed1_original_{method}_seed1_1000"
        paths.extend([
            f"results/training_metrics/{stem}.csv",
            f"results/training_metrics/{stem}_config.json",
            f"results/training_metrics/{stem}_"
            f"{'rge_diagnostics' if method.endswith('RGE') else 'rcc_diagnostics'}.jsonl",
        ])
    git("add", "--", *paths)
    git("diff", "--cached", "--check")
    git("commit", "-m", "Test RGE and coordinated coverage for FedPhoenixRG")
    sha = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    git("push", "origin", branch)
    save_status(
        status="pushed", starting_head=STARTING_HEAD,
        final_commit_sha=sha, branch=branch,
        results={
            method: {key: value for key, value in run.items() if key not in {
                "selected_clients", "task_seeds", "diagnostics"
            }} for method, run in runs.items()
        },
        rge_diagnostics=rge, rcc_diagnostics=rcc,
        summary="results/rge_rcc_1000_summary.md",
        third_or_combined_run=False,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save_status(status="failed", error=repr(error))
        raise
