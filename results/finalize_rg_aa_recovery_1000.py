"""Finalize, summarize, commit, and push the two authorized mechanism runs."""

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
STATUS = RESULTS / "rg_aa_recovery_1000_finalize_status.json"
EXPECTED_HEAD = "79270ea630cf686178da7081a22acbd1e4da07ad"
RUNS = {
    "FedPhoenixRG-AA": RESULTS / "rg_aa_1000",
    "FedPhoenixRG-Recovery": RESULTS / "rg_recovery_1000",
}


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


def assert_existing_code_unchanged():
    current_main = (ROOT / "main_fed.py").read_text(encoding="utf-8")
    head_main = run_git("show", "HEAD:main_fed.py")
    for name in (
        "FedPhoenix", "FedPhoenixRG", "FedPhoenixRGRemapped",
        "FedPhoenixRGLC", "_build_fedphoenix_tasks",
    ):
        assert ast.dump(
            function(current_main, name), include_attributes=False
        ) == ast.dump(
            function(head_main, name), include_attributes=False
        ), f"existing function changed: {name}"
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
        ), f"existing q estimator changed: {name}"


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


def parse_fields(line):
    fields = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


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
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    accuracies = [float(row["test_accuracy"]) for row in rows]
    peak_index = max(range(len(accuracies)), key=accuracies.__getitem__)
    peak = accuracies[peak_index]
    lower = max(0, peak_index - 10)
    upper = min(len(accuracies), peak_index + 11)
    neighbors = accuracies[lower:peak_index] + accuracies[peak_index + 1:upper]
    return {
        "peak": peak,
        "round": peak_index + 1,
        "near_peak_count": sum(
            value >= peak - 0.10 for value in accuracies[lower:upper]
        ),
        "best_neighbor": max(neighbors) if neighbors else float("nan"),
        "above_reference_near_peak": sum(value > 83.43 for value in accuracies[lower:upper]),
        "stdout": stdout,
        "stdout_path": stdout_path.relative_to(ROOT).as_posix(),
        "metrics_path": metrics_path.relative_to(ROOT).as_posix(),
        "config_path": config_path.relative_to(ROOT).as_posix(),
    }


def analyze_aa(run):
    rows = [
        parse_fields(line) for line in run["stdout"].splitlines()
        if line.startswith("RG_AA_SCORE ")
    ]
    if len(rows) != 650:
        raise RuntimeError(f"AA has {len(rows)} diagnostics; expected 650")
    active = [row for row in rows if row["top_reset_overlap"] != "none"]
    return {
        "diagnostic_rows": len(rows),
        "mean_abs_q_difference": statistics.mean(
            float(row["mean_abs_q_difference"]) for row in rows
        ),
        "mean_q_correlation": statistics.mean(
            float(row["correlation_q_equal_aa"]) for row in rows
        ),
        "min_q_correlation": min(
            float(row["correlation_q_equal_aa"]) for row in rows
        ),
        "mean_top_reset_overlap": statistics.mean(
            float(row["top_reset_overlap"]) for row in active
        ),
        "fraction_active_rankings_changed": statistics.mean(
            float(row["top_reset_overlap"]) < 1.0 for row in active
        ),
        "active_ranking_rows": len(active),
        "eligible_sample_min": min(
            float(row["min_eligible_client_sample_count"]) for row in rows
        ),
        "eligible_sample_max": max(
            float(row["max_eligible_client_sample_count"]) for row in rows
        ),
    }


def analyze_recovery(run):
    rows = [
        parse_fields(line) for line in run["stdout"].splitlines()
        if line.startswith("RG_RECOVERY_SCORE ")
    ]
    if len(rows) != 650:
        raise RuntimeError(
            f"Recovery has {len(rows)} diagnostics; expected 650"
        )
    fallbacks = {}
    for row in rows:
        fallbacks[row["fallback"]] = fallbacks.get(row["fallback"], 0) + 1
    return {
        "diagnostic_rows": len(rows),
        "mean_rho": statistics.mean(float(row["mean_rho"]) for row in rows),
        "mean_within_layer_rho_std": statistics.mean(
            float(row["std_rho"]) for row in rows
        ),
        "min_rho": min(float(row["min_rho"]) for row in rows),
        "max_rho": max(float(row["max_rho"]) for row in rows),
        "mean_fraction_without_observations": statistics.mean(
            float(row["fraction_without_recovery_observations"]) for row in rows
        ),
        "mean_observation_count": statistics.mean(
            float(row["mean_recovery_observation_count"]) for row in rows
        ),
        "mean_correlation_q_rho": statistics.mean(
            float(row["correlation_q_rho"]) for row in rows
        ),
        "mean_correlation_q_u": statistics.mean(
            float(row["correlation_q_u"]) for row in rows
        ),
        "min_correlation_q_u": min(
            float(row["correlation_q_u"]) for row in rows
        ),
        "fallback_counts": fallbacks,
    }


def write_summary(runs, aa, recovery):
    lines = [
        "# FedPhoenixRG mechanism experiments: 1000-round results",
        "",
        "Peak Accuracy is the sole primary selection metric. Both runs use",
        "CIFAR-10/VGG16, the fixed beta=0.3 partition, seed 1, 1000 rounds,",
        "`rg_mix=0.75`, `rg_interval=20`, `FP_conv=1000`, reset ratio 1/64,",
        "and `ori_normal`. Existing FedPhoenix and RG references were not rerun.",
        "",
        "| Method | Peak | Peak Round | vs FedPhoenix | vs 83.43% |",
        "|---|---:|---:|---:|---:|",
        "| FedPhoenix | 82.61% | 997 | — | -0.82 pp |",
        "| Current-best FedPhoenixRG | **83.43%** | 984 | +0.82 pp | — |",
    ]
    for method in RUNS:
        run = runs[method]
        lines.append(
            f"| {method} | {run['peak']:.2f}% | {run['round']} | "
            f"{run['peak'] - 82.61:+.2f} pp | {run['peak'] - 83.43:+.2f} pp |"
        )
    lines += [
        "",
        "## Aggregation-Aligned diagnostics",
        "",
        f"Across {aa['diagnostic_rows']} layer-refresh rows, the mean absolute "
        f"difference between equal-client q and AA q was "
        f"{aa['mean_abs_q_difference']:.6f}. Their mean Pearson correlation was "
        f"{aa['mean_q_correlation']:.6f} (minimum {aa['min_q_correlation']:.6f}). "
        f"Among {aa['active_ranking_rows']} active-layer refreshes, mean top-reset "
        f"overlap was {aa['mean_top_reset_overlap']:.4f}, and "
        f"{aa['fraction_active_rankings_changed']:.1%} changed at least one top-k "
        "filter. Eligible client sample counts ranged from "
        f"{aa['eligible_sample_min']:.0f} to {aa['eligible_sample_max']:.0f}.",
        "",
        "## Reset Recoverability diagnostics",
        "",
        f"Across {recovery['diagnostic_rows']} layer-refresh rows, mean rho was "
        f"{recovery['mean_rho']:.6f}; mean within-layer rho standard deviation was "
        f"{recovery['mean_within_layer_rho_std']:.6f}, with observed range "
        f"{recovery['min_rho']:.6f}–{recovery['max_rho']:.6f}. The mean fraction "
        "of filters without a recovery observation was "
        f"{recovery['mean_fraction_without_observations']:.1%}, and the mean raw "
        f"observation count per filter was {recovery['mean_observation_count']:.3f}. "
        f"Mean correlation(q, rho) was {recovery['mean_correlation_q_rho']:.6f}; "
        f"mean correlation(q, u) was {recovery['mean_correlation_q_u']:.6f} "
        f"(minimum {recovery['min_correlation_q_u']:.6f}). Fallback counts were "
        f"{json.dumps(recovery['fallback_counts'], sort_keys=True)}.",
        "",
        "## Peak decision",
        "",
    ]
    winners = [method for method in RUNS if runs[method]["peak"] > 83.43]
    if not winners:
        lines.append(
            "Neither method exceeded 83.43%. The reset-placement optimization "
            "route stops here, and no additional variant was started."
        )
    else:
        for method in winners:
            run = runs[method]
            delta = run["peak"] - 83.43
            if delta <= 0.10:
                judgment = "a small gain that is reported without declaring a new best"
            elif run["above_reference_near_peak"] > 1:
                judgment = "supported by more than one above-reference point nearby"
            else:
                judgment = "an isolated above-reference high point"
            lines.append(
                f"{method} exceeded 83.43% by {delta:.2f} pp; this is {judgment}. "
                f"Within ±10 rounds, {run['above_reference_near_peak']} points "
                f"exceeded 83.43%, and the best neighboring accuracy was "
                f"{run['best_neighbor']:.2f}%. No additional experiment was started."
            )
    lines += [
        "",
        "Both runs completed 1000 rounds with exit code 0. Complete stdout,",
        "stderr, manifests, round metrics, configs, and per-filter recovery",
        "observation counts are saved under `results/`.",
    ]
    (RESULTS / "rg_aa_recovery_1000_summary.md").write_text(
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
    runs = {
        method: load_run(method, manifests[method][0]) for method in RUNS
    }
    aa = analyze_aa(runs["FedPhoenixRG-AA"])
    recovery = analyze_recovery(runs["FedPhoenixRG-Recovery"])
    if run_git("rev-parse", "HEAD") != EXPECTED_HEAD:
        raise RuntimeError("repository HEAD changed during training; manual review needed")
    if run_git("branch", "--show-current") != "main":
        raise RuntimeError("branch changed during training; manual review needed")
    if run_git("diff", "--cached", "--name-only"):
        raise RuntimeError("index has unrelated staged files; manual review needed")
    assert_existing_code_unchanged()
    run_git("diff", "--check")
    write_summary(runs, aa, recovery)
    paths = [
        "Algorithm/repeatability_guidance.py", "main_fed.py",
        "train_five_baselines.py", "tests/test_repeatability_guidance.py",
        "tests/test_fedphoenix_rg.py",
        "results/finalize_rg_aa_recovery_1000.py",
        "results/rg_aa_recovery_1000_summary.md",
        "results/rg_aa_1000", "results/rg_recovery_1000",
    ]
    for method in RUNS:
        stem = f"cifar10_vgg_{method}_seed1_original_{method}_seed1_1000"
        paths.extend([
            f"results/training_metrics/{stem}.csv",
            f"results/training_metrics/{stem}_config.json",
        ])
    run_git("add", "--", *paths)
    run_git("diff", "--cached", "--check")
    run_git("commit", "-m", "Add aggregation-aligned and recovery RG experiments")
    sha = run_git("rev-parse", "HEAD")
    run_git("push", "origin", "main")
    save_status(
        status="pushed", commit_sha=sha,
        results={
            method: {
                key: value for key, value in run.items() if key != "stdout"
            } for method, run in runs.items()
        },
        aa_diagnostics=aa,
        recovery_diagnostics=recovery,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save_status(status="failed", error=str(error))
        raise
