"""Finalize the two selected intervention-cycle experiments after completion."""

from __future__ import annotations

import ast
import csv
import json
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
STATUS = RESULTS / "remaining_mechanisms_1000_finalize_status.json"
STARTING_HEAD = "3529931f35cd50f37fdd824832e99a1d6f73365b"
RUNS = {
    "FedPhoenixRG-DeltaAgg": RESULTS / "rg_deltaagg_1000",
    "FedPhoenixRG-Permute": RESULTS / "rg_permute_1000",
}
EXPECTED_MECHANISM_ROWS = 7006


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


def parse_fields(line):
    fields = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def function(source, name):
    return next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def class_method(tree, class_name, method_name):
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def assert_frozen_code_unchanged():
    current_main = (ROOT / "main_fed.py").read_text(encoding="utf-8")
    head_main = run_git("show", "HEAD:main_fed.py")
    for name in (
        "FedPhoenix", "FedPhoenixRG", "FedPhoenixRGRemapped",
        "FedPhoenixRGLC", "FedPhoenixRGAA", "FedPhoenixRGRecovery",
        "_build_fedphoenix_tasks",
    ):
        assert ast.dump(
            function(current_main, name), include_attributes=False
        ) == ast.dump(
            function(head_main, name), include_attributes=False
        ), f"frozen function changed: {name}"
    scorer_path = "Algorithm/repeatability_guidance.py"
    current_scorer = ast.parse((ROOT / scorer_path).read_text(encoding="utf-8"))
    head_scorer = ast.parse(run_git("show", f"HEAD:{scorer_path}"))
    for name in ("observe_round", "finalize_window", "reset_window"):
        assert ast.dump(
            class_method(current_scorer, "RepeatabilityGuidance", name),
            include_attributes=False,
        ) == ast.dump(
            class_method(head_scorer, "RepeatabilityGuidance", name),
            include_attributes=False,
        ), f"frozen q estimator changed: {name}"
    changed = set(run_git("diff", "--name-only").splitlines())
    forbidden = {
        "Algorithm/Phoenix_util.py", "models/Fed.py", "models/Update.py",
        "Algorithm/repeatability_guidance.py",
    }
    if changed & forbidden:
        raise RuntimeError(f"frozen supporting code changed: {changed & forbidden}")


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


def load_run(method, manifest_path):
    stem = f"cifar10_vgg_{method}_seed1_original_{method}_seed1_1000"
    metrics_path = RESULTS / "training_metrics" / f"{stem}.csv"
    config_path = RESULTS / "training_metrics" / f"{stem}_config.json"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1000 or [int(row["round"]) for row in rows] != list(range(1, 1001)):
        raise RuntimeError(f"{method} metrics are not 1000 consecutive rounds")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "epochs": 1000, "seed": 1, "rg_mix": 0.75, "rg_interval": 20,
        "FP_conv": 1000, "FP_fc": 0, "reset": 0.015625,
        "remethod": "ori_normal", "num_users": 100, "frac": 0.1,
        "local_ep": 5, "local_bs": 50, "lr": 0.01, "momentum": 0.5,
        "weight_decay": 0.0, "dataset": "cifar10", "model": "vgg",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise RuntimeError(f"{method} config mismatch: {key}={config[key]}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stdout_path = Path(manifest["runs"][0]["stdout"])
    stderr_path = Path(manifest["runs"][0]["stderr"])
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    accuracies = [float(row["test_accuracy"]) for row in rows]
    peak_index = max(range(len(accuracies)), key=accuracies.__getitem__)
    lo = max(0, peak_index - 5)
    hi = min(len(accuracies), peak_index + 6)
    neighborhood = accuracies[lo:hi]
    return {
        "peak": accuracies[peak_index],
        "peak_round": peak_index + 1,
        "round_1000": accuracies[-1],
        "neighborhood_start": lo + 1,
        "neighborhood_end": hi,
        "neighborhood_mean": statistics.mean(neighborhood),
        "neighborhood_second_best": sorted(neighborhood, reverse=True)[1],
        "neighborhood_above_reference": sum(value > 83.43 for value in neighborhood),
        "selected_clients": [row["selected_clients"] for row in rows],
        "task_seeds": [row["task_seeds"] for row in rows],
        "stdout": stdout,
        "stdout_path": stdout_path.relative_to(ROOT).as_posix(),
        "stderr_path": stderr_path.relative_to(ROOT).as_posix(),
        "metrics_path": metrics_path.relative_to(ROOT).as_posix(),
        "config_path": config_path.relative_to(ROOT).as_posix(),
    }


def weighted_mean(rows, value_key, weight_key="reset_observations"):
    total = sum(float(row[weight_key]) for row in rows)
    return sum(
        float(row[value_key]) * float(row[weight_key]) for row in rows
    ) / total


def analyze_delta(run):
    rows = [
        parse_fields(line) for line in run["stdout"].splitlines()
        if line.startswith("RG_DELTA_AGG ")
    ]
    if len(rows) != EXPECTED_MECHANISM_ROWS:
        raise RuntimeError(
            f"DeltaAgg has {len(rows)} diagnostics; expected {EXPECTED_MECHANISM_ROWS}"
        )
    return {
        "diagnostic_rows": len(rows),
        "reset_observations": sum(int(row["reset_observations"]) for row in rows),
        "mean_perturbation_norm": weighted_mean(rows, "mean_perturbation_norm"),
        "mean_local_step_norm": weighted_mean(rows, "mean_local_step_norm"),
        "mean_step_over_perturbation": weighted_mean(
            rows, "mean_step_over_perturbation"
        ),
        "mean_perturbation_step_cosine": weighted_mean(
            rows, "mean_perturbation_step_cosine"
        ),
        "mean_aggregate_correction_norm": statistics.mean(
            float(row["aggregate_correction_norm"]) for row in rows
        ),
        "mean_multi_client_filter_fraction": statistics.mean(
            float(row["multi_client_filter_fraction"]) for row in rows
        ),
    }


def analyze_permutation(run):
    rows = [
        parse_fields(line) for line in run["stdout"].splitlines()
        if line.startswith("RG_PERMUTE_DIAG ")
    ]
    if len(rows) != EXPECTED_MECHANISM_ROWS:
        raise RuntimeError(
            f"Permute has {len(rows)} diagnostics; expected {EXPECTED_MECHANISM_ROWS}"
        )
    return {
        "diagnostic_rows": len(rows),
        "reset_observations": sum(int(row["reset_observations"]) for row in rows),
        "mean_relative_perturbation_norm": weighted_mean(
            rows, "mean_relative_perturbation_norm"
        ),
        "mean_original_permuted_cosine": weighted_mean(
            rows, "mean_original_permuted_cosine"
        ),
        "mean_fixed_position_fraction": weighted_mean(
            rows, "mean_fixed_position_fraction"
        ),
    }


def verify_shared_sequences(runs):
    reference_path = RESULTS / "training_metrics" / (
        "cifar10_vgg_FedPhoenixRG_seed1_paired1200_rg_mix075.csv"
    )
    with reference_path.open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))[:1000]
    reference_clients = [row["selected_clients"] for row in reference]
    reference_seeds = [row["task_seeds"] for row in reference]
    for method, run in runs.items():
        if run["selected_clients"] != reference_clients:
            raise RuntimeError(f"{method} client sampling sequence changed")
        if run["task_seeds"] != reference_seeds:
            raise RuntimeError(f"{method} task seed sequence changed")


def decision_text(method, run):
    delta = run["peak"] - 83.43
    if delta <= 0:
        return (
            f"{method} did not exceed 83.43%; its mechanism hypothesis is weakened."
        )
    if delta <= 0.10:
        return (
            f"{method} improved by {delta:.2f} pp, classified as a marginal "
            "single-seed improvement rather than a new best."
        )
    if run["peak"] >= 83.70 and run["neighborhood_above_reference"] > 1:
        return (
            f"{method} exceeded 83.43% by {delta:.2f} pp with "
            f"{run['neighborhood_above_reference']} above-reference points in "
            "the ±5 neighborhood; it is worth further analysis."
        )
    return (
        f"{method} exceeded 83.43% by {delta:.2f} pp, but neighborhood support "
        "is insufficient for a strong claim."
    )


def write_summary(runs, delta_diag, permute_diag):
    lines = [
        "# FedPhoenixRG remaining-mechanism experiments",
        "",
        f"Starting HEAD: `{STARTING_HEAD}` on `main`.",
        "",
        "## Research selection",
        "",
        "Placement was deprioritized because current-best RG already gives",
        "positive q to 98.9% of active filters with only a 0.0268-nat mean",
        "entropy gap, while LC, Excess, Persistent, AA, and q×recovery all",
        "failed to improve Peak. Four candidate mechanism questions were scored:",
        "",
        "| Rank | Candidate | Evidence | Orthogonality | Upside | Interpretability | Simplicity | Fair compute | Negative value | Risk |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        "| 1 | Actual-task delta aggregation | 5 | 5 | 4 | 5 | 4 | 5 | 5 | Medium |",
        "| 2 | Within-filter permutation | 4 | 5 | 4 | 5 | 4 | 5 | 5 | Medium |",
        "| 3 | One-epoch reset-only warmup | 3 | 5 | 3 | 4 | 3 | 4 | 4 | High |",
        "| 4 | State-aware intervention timing | 2 | 4 | 3 | 3 | 2 | 5 | 3 | High |",
        "",
        "The first two were selected because they test independent aggregation",
        "and perturbation bottlenecks with one discrete change each. Warmup was",
        "rejected because it removes one fifth of ordinary parameter updates;",
        "timing was rejected because existing diagnostics provide no direct",
        "per-round intervention utility signal.",
        "",
        "## Results",
        "",
        "| Method | Peak | Peak round | vs FedPhoenix | vs FedPhoenixRG | Round 1000 | Peak-neighborhood mean | Neighborhood second-best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| FedPhoenix | 82.61% | 997 | — | -0.82 pp | 80.59% | — | — |",
        "| Current-best FedPhoenixRG | **83.43%** | 984 | +0.82 pp | — | 81.27% | 81.84% | 83.39% |",
    ]
    for method in RUNS:
        run = runs[method]
        lines.append(
            f"| {method} | {run['peak']:.2f}% | {run['peak_round']} | "
            f"{run['peak'] - 82.61:+.2f} pp | {run['peak'] - 83.43:+.2f} pp | "
            f"{run['round_1000']:.2f}% | {run['neighborhood_mean']:.2f}% | "
            f"{run['neighborhood_second_best']:.2f}% |"
        )
    lines += [
        "",
        "Peak neighborhoods are clipped to rounds 1–1000 and otherwise cover",
        "peak-5 through peak+5.",
        "",
        "## FedPhoenixRG-DeltaAgg",
        "",
        "Scientific question: does ordinary absolute-state averaging confuse",
        "the random reset start with a learned client update? Reset filters",
        "contribute `theta_global + (theta_local - theta_reset)` at their original",
        "sample weight; every other parameter uses ordinary FedAvg.",
        "",
        f"Across {delta_diag['diagnostic_rows']} layer-round diagnostics and "
        f"{delta_diag['reset_observations']} reset observations, mean perturbation "
        f"norm was {delta_diag['mean_perturbation_norm']:.6f}, mean local-step norm "
        f"was {delta_diag['mean_local_step_norm']:.6f}, and their mean ratio was "
        f"{delta_diag['mean_step_over_perturbation']:.6f}. Mean perturbation/local-"
        f"step cosine was {delta_diag['mean_perturbation_step_cosine']:.6f}; mean "
        f"aggregate correction norm was {delta_diag['mean_aggregate_correction_norm']:.6f}. "
        f"The mean fraction of reset filters hit by multiple clients in a round "
        f"was {delta_diag['mean_multi_client_filter_fraction']:.1%}.",
        "",
        decision_text("FedPhoenixRG-DeltaAgg", runs["FedPhoenixRG-DeltaAgg"]),
        "",
        "## FedPhoenixRG-Permute",
        "",
        "Scientific question: does full layer-distribution resampling destroy",
        "too much filter-specific information? Selected indices use the same RG",
        "sampler, but task-private permutations preserve each selected filter's",
        "exact values, mean, variance, and norm while disrupting arrangement.",
        "",
        f"Across {permute_diag['diagnostic_rows']} layer-round diagnostics and "
        f"{permute_diag['reset_observations']} reset observations, mean relative "
        f"perturbation norm was {permute_diag['mean_relative_perturbation_norm']:.6f}, "
        f"mean original/permuted cosine was "
        f"{permute_diag['mean_original_permuted_cosine']:.6f}, and the mean fraction "
        f"of scalar positions left fixed was "
        f"{permute_diag['mean_fixed_position_fraction']:.6f}.",
        "",
        decision_text("FedPhoenixRG-Permute", runs["FedPhoenixRG-Permute"]),
        "",
        "## Overall interpretation",
        "",
    ]
    if all(run["peak"] <= 83.43 for run in runs.values()):
        lines.append(
            "Both independent mechanism tests failed to exceed current-best RG. "
            "Together with the prior placement negatives, this supports the "
            "conclusion that FedPhoenixRG is likely near a local performance "
            "ceiling under the current fixed framework. No follow-up was started."
        )
    else:
        lines.append(
            "At least one mechanism exceeded current-best RG; the classification",
        )
        lines.append(
            "above determines whether the result is marginal or worth further",
        )
        lines.append(
            "analysis. No follow-up was started automatically."
        )
    lines += [
        "",
        "Original FedPhoenixRG behavior remained unchanged. No hyperparameter",
        "sweep, combined variant, or unplanned follow-up method was run.",
        "",
        "Complete stdout/stderr, commands, manifests, metrics, configs, and",
        "mechanism diagnostics are stored in `results/rg_deltaagg_1000/`,",
        "`results/rg_permute_1000/`, and `results/training_metrics/`.",
    ]
    (RESULTS / "remaining_mechanisms_1000_summary.md").write_text(
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
        raise RuntimeError("timed out waiting for both formal runs")
    runs = {
        method: load_run(method, manifests[method][0]) for method in RUNS
    }
    verify_shared_sequences(runs)
    delta_diag = analyze_delta(runs["FedPhoenixRG-DeltaAgg"])
    permute_diag = analyze_permutation(runs["FedPhoenixRG-Permute"])
    if run_git("rev-parse", "HEAD") != STARTING_HEAD:
        raise RuntimeError("repository HEAD changed during training; manual review needed")
    if run_git("branch", "--show-current") != "main":
        raise RuntimeError("branch changed during training; manual review needed")
    if run_git("diff", "--cached", "--name-only"):
        raise RuntimeError("index has unrelated staged files; manual review needed")
    assert_frozen_code_unchanged()
    run_git("diff", "--check")
    write_summary(runs, delta_diag, permute_diag)
    paths = [
        "Algorithm/intervention_variants.py", "main_fed.py",
        "train_five_baselines.py", "tests/test_intervention_variants.py",
        "tests/test_fedphoenix_rg.py", "results/mechanism_candidate_assessment.md",
        "results/finalize_remaining_mechanisms_1000.py",
        "results/remaining_mechanisms_1000_summary.md",
        "results/rg_deltaagg_1000", "results/rg_permute_1000",
    ]
    for method in RUNS:
        stem = f"cifar10_vgg_{method}_seed1_original_{method}_seed1_1000"
        paths.extend([
            f"results/training_metrics/{stem}.csv",
            f"results/training_metrics/{stem}_config.json",
        ])
    run_git("add", "--", *paths)
    run_git("diff", "--cached", "--check")
    run_git("commit", "-m", "Test remaining FedPhoenixRG intervention mechanisms")
    sha = run_git("rev-parse", "HEAD")
    branch = run_git("branch", "--show-current")
    run_git("push", "origin", branch)
    save_status(
        status="pushed", starting_head=STARTING_HEAD,
        commit_sha=sha, branch=branch,
        results={
            method: {
                key: value for key, value in run.items()
                if key not in {"stdout", "selected_clients", "task_seeds"}
            } for method, run in runs.items()
        },
        delta_diagnostics=delta_diag,
        permutation_diagnostics=permute_diag,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save_status(status="failed", error=str(error))
        raise
