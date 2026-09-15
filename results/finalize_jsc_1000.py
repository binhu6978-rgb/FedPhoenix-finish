"""Validate, summarize, commit, and push the single JSC formal run."""

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
STATUS = RESULTS / "jsc_1000_finalize_status.json"
METHOD = "FedPhoenixRG-JSC"
STARTING_HEAD = "c7152f014264f3f6306d87e6705c63dbbd144280"


def save_status(**fields):
    STATUS.write_text(
        json.dumps(
            {"updated_at": datetime.now().isoformat(), **fields},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8",
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
        ), f"frozen q behavior changed: {name}"


def load_run():
    manifests = sorted((RESULTS / "rg_jsc_1000").glob("*/manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"expected exactly one formal manifest, found {len(manifests)}")
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("formal_experiment_count") != 1 or len(manifest["runs"]) != 1:
        raise RuntimeError("formal manifest contains an unexpected extra run")
    run_manifest = manifest["runs"][0]
    if (
        manifest.get("status") != "complete"
        or run_manifest.get("method") != METHOD
        or run_manifest.get("exit_code") != 0
    ):
        raise RuntimeError("JSC formal run did not finish successfully")

    stem = f"cifar10_vgg_{METHOD}_seed1_original_{METHOD}_seed1_1000"
    metrics_path = RESULTS / "training_metrics" / f"{stem}.csv"
    config_path = RESULTS / "training_metrics" / f"{stem}_config.json"
    diagnostics_path = RESULTS / "training_metrics" / f"{stem}_jsc_diagnostics.jsonl"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1000 or [int(row["round"]) for row in rows] != list(range(1, 1001)):
        raise RuntimeError("formal metrics are not exactly rounds 1 through 1000")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "algorithm": METHOD, "epochs": 1000, "seed": 1,
        "dataset": "cifar10", "model": "vgg", "num_users": 100,
        "frac": 0.1, "local_ep": 5, "local_bs": 50,
        "lr": 0.01, "momentum": 0.5, "weight_decay": 0.0,
        "rg_mix": 0.75, "rg_interval": 20, "reset": 0.015625,
        "remethod": "ori_normal", "FP_conv": 1000, "FP_fc": 0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"config mismatch: {key}={config.get(key)}")
    with diagnostics_path.open(encoding="utf-8") as handle:
        diagnostics = [json.loads(line) for line in handle if line.strip()]
    if len(diagnostics) != 650:
        raise RuntimeError(f"expected 650 layer-refresh diagnostics, found {len(diagnostics)}")

    accuracies = [float(row["test_accuracy"]) for row in rows]
    peak_index = max(range(1000), key=accuracies.__getitem__)
    start = max(0, peak_index - 5)
    end = min(1000, peak_index + 6)
    neighborhood = accuracies[start:end]
    return {
        "peak": accuracies[peak_index], "peak_round": peak_index + 1,
        "round_1000": accuracies[-1],
        "neighborhood_start": start + 1, "neighborhood_end": end,
        "neighborhood_mean": statistics.mean(neighborhood),
        "neighborhood_second_best": sorted(neighborhood, reverse=True)[1],
        "selected_clients": [row["selected_clients"] for row in rows],
        "task_seeds": [row["task_seeds"] for row in rows],
        "accuracies": accuracies, "diagnostics": diagnostics,
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "metrics_path": metrics_path.relative_to(ROOT).as_posix(),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "diagnostics_path": diagnostics_path.relative_to(ROOT).as_posix(),
        "stdout_path": Path(run_manifest["stdout"]).relative_to(ROOT).as_posix(),
        "stderr_path": Path(run_manifest["stderr"]).relative_to(ROOT).as_posix(),
    }


def verify_first_twenty(run):
    reference_path = RESULTS / "training_metrics" / (
        "cifar10_vgg_FedPhoenixRG_seed1_paired1200_rg_mix075.csv"
    )
    with reference_path.open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))[:20]
    if run["selected_clients"][:20] != [row["selected_clients"] for row in reference]:
        raise RuntimeError("formal first-20 client selection differs from current-best")
    if run["task_seeds"][:20] != [row["task_seeds"] for row in reference]:
        raise RuntimeError("formal first-20 task seeds differ from current-best")
    reference_accuracy = [float(row["test_accuracy"]) for row in reference]
    if run["accuracies"][:20] != reference_accuracy:
        raise RuntimeError("formal first-20 accuracies differ from current-best")


def weighted_mean(rows, key, weight_key):
    total = sum(float(row[weight_key]) for row in rows)
    return sum(float(row[key]) * float(row[weight_key]) for row in rows) / total


def weighted_median(rows, key, weight_key):
    ordered = sorted((float(row[key]), float(row[weight_key])) for row in rows)
    halfway = sum(weight for _value, weight in ordered) / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= halfway:
            return value
    return ordered[-1][0]


def summarize_diagnostics(rows):
    active = [row for row in rows if row["active_rounds_until_next_refresh"] > 0]
    for row in active:
        row["filter_round_weight"] = (
            row["num_filters"] * row["active_rounds_until_next_refresh"]
        )
        row["layer_round_weight"] = row["active_rounds_until_next_refresh"]
    filter_weight = "filter_round_weight"
    layer_weight = "layer_round_weight"
    return {
        "refresh_layer_rows": len(rows),
        "active_refresh_layer_rows": len(active),
        "mean_jackknife_se": weighted_mean(active, "mean_jackknife_se", filter_weight),
        "median_jackknife_se": weighted_median(active, "median_jackknife_se", filter_weight),
        "max_jackknife_se": max(row["max_jackknife_se"] for row in active),
        "mean_within_layer_se_std": weighted_mean(active, "within_layer_se_std", filter_weight),
        "mean_q_minus_s": weighted_mean(active, "mean_q_minus_s", filter_weight),
        "fraction_s_less_than_q": weighted_mean(active, "fraction_s_less_than_q", filter_weight),
        "fraction_s_zero": weighted_mean(active, "fraction_s_zero", filter_weight),
        "pearson_q_vs_s": weighted_mean(active, "pearson_q_vs_s", filter_weight),
        "spearman_q_vs_s": weighted_mean(active, "spearman_q_vs_s", filter_weight),
        "top_reset_count_overlap": weighted_mean(active, "top_reset_count_overlap", layer_weight),
        "mean_eligible_clients": weighted_mean(active, "mean_eligible_clients", filter_weight),
        "min_eligible_clients": min(row["min_eligible_clients"] for row in active),
        "max_eligible_clients": max(row["max_eligible_clients"] for row in active),
        "corr_se_vs_eligible_clients": weighted_mean(active, "corr_se_vs_eligible_clients", filter_weight),
        "corr_se_vs_q": weighted_mean(active, "corr_se_vs_q", filter_weight),
        "raw_q_sampling_entropy": weighted_mean(active, "raw_q_sampling_entropy", layer_weight),
        "jsc_sampling_entropy": weighted_mean(active, "jsc_sampling_entropy", layer_weight),
        "raw_q_entropy_gap_from_uniform": weighted_mean(active, "raw_q_entropy_gap_from_uniform", layer_weight),
        "jsc_entropy_gap_from_uniform": weighted_mean(active, "jsc_entropy_gap_from_uniform", layer_weight),
        "raw_q_max_probability_over_uniform": weighted_mean(active, "raw_q_max_probability_over_uniform", layer_weight),
        "jsc_max_probability_over_uniform": weighted_mean(active, "jsc_max_probability_over_uniform", layer_weight),
    }


def write_summary(run, diag):
    delta_rg = run["peak"] - 83.43
    if run["peak"] <= 83.43:
        interpretation = (
            "The fixed one-jackknife-SE calibration did not improve current-best RG."
        )
    elif run["peak"] < 83.70:
        interpretation = (
            "The result is a marginal single-seed improvement and is not treated as "
            "a clear success."
        )
    else:
        interpretation = (
            "The peak reached the approximately 83.7% threshold; its neighborhood "
            "must also improve before further study is justified."
        )
    lines = [
        "# FedPhoenixRG-JSC 1000-round experiment", "",
        f"Starting HEAD: `{STARTING_HEAD}` on `main`.", "",
        "## Result", "",
        "| Method | Peak | Peak round | vs FedPhoenix | vs FedPhoenixRG | Round 1000 | Peak ±5 rounds | Neighborhood mean | Neighborhood second-best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| FedPhoenix | 82.61% | 997 | — | -0.82 pp | 80.59% | — | — | — |",
        "| Current-best FedPhoenixRG | **83.43%** | 984 | +0.82 pp | — | 81.27% | 979–989 | 81.84% | 83.39% |",
        f"| FedPhoenixRG-JSC | {run['peak']:.2f}% | {run['peak_round']} | "
        f"{run['peak'] - 82.61:+.2f} pp | {delta_rg:+.2f} pp | "
        f"{run['round_1000']:.2f}% | {run['neighborhood_start']}–"
        f"{run['neighborhood_end']} | {run['neighborhood_mean']:.2f}% | "
        f"{run['neighborhood_second_best']:.2f}% |", "",
        "## Mechanism diagnostics", "",
        "Statistics below are weighted by the active filter-rounds or active "
        "layer-rounds on which each refreshed distribution could affect training.", "",
        f"- Jackknife SE: mean {diag['mean_jackknife_se']:.6f}, median-of-layer/filter medians "
        f"{diag['median_jackknife_se']:.6f}, maximum {diag['max_jackknife_se']:.6f}.",
        f"- Mean q-s: {diag['mean_q_minus_s']:.6f}; fraction s<q: "
        f"{diag['fraction_s_less_than_q']:.2%}; fraction s=0: {diag['fraction_s_zero']:.2%}.",
        f"- q versus s: Pearson {diag['pearson_q_vs_s']:.6f}, Spearman "
        f"{diag['spearman_q_vs_s']:.6f}; top-reset-count overlap "
        f"{diag['top_reset_count_overlap']:.2%}.",
        f"- Eligible clients: mean {diag['mean_eligible_clients']:.2f}, observed "
        f"min {diag['min_eligible_clients']}, max {diag['max_eligible_clients']}.",
        f"- corr(SE, eligible clients): {diag['corr_se_vs_eligible_clients']:.6f}; "
        f"corr(SE, q): {diag['corr_se_vs_q']:.6f}; mean within-layer SE "
        f"dispersion: {diag['mean_within_layer_se_std']:.6f}.",
        f"- Sampling entropy, raw q versus JSC: {diag['raw_q_sampling_entropy']:.6f} "
        f"versus {diag['jsc_sampling_entropy']:.6f}; entropy gap from uniform: "
        f"{diag['raw_q_entropy_gap_from_uniform']:.6f} versus "
        f"{diag['jsc_entropy_gap_from_uniform']:.6f}.",
        f"- Mean max-probability/uniform ratio, raw q versus JSC: "
        f"{diag['raw_q_max_probability_over_uniform']:.6f} versus "
        f"{diag['jsc_max_probability_over_uniform']:.6f}.", "",
        "## Interpretation", "", interpretation, "",
        "JSC measures leave-one-eligible-client-out sensitivity of the unchanged "
        "client-balanced repeatability estimate. It is not interpreted as reset "
        "utility or as a frequentist confidence interval.", "",
        "The original FedPhoenix and FedPhoenixRG function bodies, raw q estimator, "
        "schedule, perturbation, independent task sampling, aggregation, and local "
        "training remained unchanged. Exactly one new 1000-round formal experiment "
        "was run. No sweep or follow-up was launched.", "",
        "Pre-run validation: 26 focused tests passed, and the 20-round validation "
        "matched current-best RG in selected clients, task seeds, and every accuracy.",
    ]
    (RESULTS / "jsc_1000_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    save_status(status="validating")
    if git("rev-parse", "HEAD") != STARTING_HEAD:
        raise RuntimeError("HEAD changed during formal training")
    if git("branch", "--show-current") != "main":
        raise RuntimeError("branch changed during formal training")
    if git("diff", "--cached", "--name-only"):
        raise RuntimeError("index contains unrelated staged files")
    run = load_run()
    verify_first_twenty(run)
    assert_frozen_behavior()
    diagnostics = summarize_diagnostics(run["diagnostics"])
    git("diff", "--check")
    write_summary(run, diagnostics)
    paths = [
        "Algorithm/repeatability_guidance.py", "main_fed.py",
        "train_five_baselines.py", "tests/test_jackknife_stability.py",
        "results/run_jsc_1000.py", "results/finalize_jsc_1000.py",
        "results/prepare_and_run_jsc.py",
        "results/jsc_first20_validation/comparison.json",
        "results/jsc_1000_summary.md", "results/rg_jsc_1000",
        run["metrics_path"], run["config_path"], run["diagnostics_path"],
    ]
    git("add", "--", *paths)
    git("diff", "--cached", "--check")
    git("commit", "-m", "Test jackknife-stabilized FedPhoenixRG guidance")
    sha = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    git("push", "origin", branch)
    save_status(
        status="pushed", starting_head=STARTING_HEAD,
        final_commit_sha=sha, branch=branch,
        result={key: value for key, value in run.items() if key not in {
            "selected_clients", "task_seeds", "accuracies", "diagnostics"
        }},
        mechanism_diagnostics=diagnostics,
        summary="results/jsc_1000_summary.md",
        exactly_one_formal_run=True, hyperparameter_sweep=False,
        unplanned_followup=False,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save_status(status="failed", error=repr(error))
        raise
