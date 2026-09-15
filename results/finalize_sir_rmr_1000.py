"""Validate, summarize, commit, and push the SIR/RMR formal runs."""

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
STATUS = RESULTS / "sir_rmr_1000_finalize_status.json"
METHODS = ("FedPhoenixRG-SIR", "FedPhoenixRG-RMR")
RUN_ROOTS = {
    "FedPhoenixRG-SIR": RESULTS / "rg_sir_1000",
    "FedPhoenixRG-RMR": RESULTS / "rg_rmr_1000",
}
STARTING_HEAD = "82b53c0a0f3c7b1d7f30d4442dcb36901a1f4ea8"


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
        ), f"raw q behavior changed: {name}"


def load_run(method):
    manifests = sorted(RUN_ROOTS[method].glob("*/manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"expected one {method} manifest, found {len(manifests)}")
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    run_manifest = manifest["runs"][0]
    if (
        manifest.get("formal_experiment_count") != 1
        or len(manifest["runs"]) != 1
        or manifest.get("status") != "complete"
        or run_manifest.get("method") != method
        or run_manifest.get("exit_code") != 0
    ):
        raise RuntimeError(f"invalid formal run manifest: {manifest_path}")
    stem = f"cifar10_vgg_{method}_seed1_original_{method}_seed1_1000"
    metrics_path = RESULTS / "training_metrics" / f"{stem}.csv"
    config_path = RESULTS / "training_metrics" / f"{stem}_config.json"
    suffix = "sir" if method.endswith("SIR") else "rmr"
    diagnostics_path = RESULTS / "training_metrics" / f"{stem}_{suffix}_diagnostics.jsonl"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1000 or [int(row["round"]) for row in rows] != list(range(1, 1001)):
        raise RuntimeError(f"{method} does not contain exactly rounds 1-1000")
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
    with diagnostics_path.open(encoding="utf-8") as handle:
        diagnostics = [json.loads(line) for line in handle if line.strip()]
    if len(diagnostics) != 343:
        raise RuntimeError(f"{method} expected 343 active diagnostics, found {len(diagnostics)}")
    spectrum_error = max(
        row["max_sorted_probability_absolute_difference"] for row in diagnostics
    )
    if spectrum_error > 1e-15:
        raise RuntimeError(f"{method} probability spectrum changed by {spectrum_error}")
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


def verify_first_twenty(runs):
    reference_path = RESULTS / "training_metrics" / (
        "cifar10_vgg_FedPhoenixRG_seed1_paired1200_rg_mix075.csv"
    )
    with reference_path.open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))[:20]
    clients = [row["selected_clients"] for row in reference]
    seeds = [row["task_seeds"] for row in reference]
    accuracies = [float(row["test_accuracy"]) for row in reference]
    for method, run in runs.items():
        if run["selected_clients"][:20] != clients:
            raise RuntimeError(f"{method} first-20 clients changed")
        if run["task_seeds"][:20] != seeds:
            raise RuntimeError(f"{method} first-20 task seeds changed")
        if run["accuracies"][:20] != accuracies:
            raise RuntimeError(f"{method} first-20 accuracies changed")


def means(rows, keys):
    return {key: statistics.mean(float(row[key]) for row in rows) for key in keys}


COMMON_KEYS = (
    "pearson_raw_q_vs_alternative", "spearman_raw_q_vs_alternative",
    "top_reset_count_overlap", "fraction_ranking_positions_changed",
    "raw_probability_entropy", "new_probability_entropy",
    "raw_entropy_gap_from_uniform", "new_entropy_gap_from_uniform",
    "raw_max_probability_over_uniform", "new_max_probability_over_uniform",
)


def summarize_sir(rows):
    result = means(rows, COMMON_KEYS + (
        "mean_jackknife_se", "median_jackknife_se", "corr_se_vs_q",
    ))
    result["max_jackknife_se"] = max(row["max_jackknife_se"] for row in rows)
    result["max_spectrum_difference"] = max(
        row["max_sorted_probability_absolute_difference"] for row in rows
    )
    return result


def summarize_rmr(rows):
    result = means(rows, COMMON_KEYS + (
        "pearson_q_vs_positive_r", "spearman_q_vs_positive_r",
        "corr_q_vs_energy", "corr_positive_r_vs_energy",
        "raw_top_mean_energy", "raw_top_mean_positive_r", "raw_top_mean_q",
        "rmr_top_mean_energy", "rmr_top_mean_positive_r", "rmr_top_mean_q",
    ))
    result["max_spectrum_difference"] = max(
        row["max_sorted_probability_absolute_difference"] for row in rows
    )
    return result


def result_row(method, run):
    return (
        f"| {method} | {run['peak']:.2f}% | {run['peak_round']} | "
        f"{run['peak'] - 82.61:+.2f} pp | {run['peak'] - 83.43:+.2f} pp | "
        f"{run['round_1000']:.2f}% | {run['neighborhood_start']}–"
        f"{run['neighborhood_end']} | {run['neighborhood_mean']:.2f}% | "
        f"{run['neighborhood_second_best']:.2f}% |"
    )


def write_summary(runs, sir, rmr):
    lines = [
        "# FedPhoenixRG spectrum-preserving reranking experiments", "",
        f"Starting HEAD: `{STARTING_HEAD}` on `main`.", "",
        "## Results", "",
        "| Method | Peak | Peak round | vs FedPhoenix | vs FedPhoenixRG | Round 1000 | Peak ±5 rounds | Neighborhood mean | Neighborhood second-best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| FedPhoenix | 82.61% | 997 | — | -0.82 pp | 80.59% | — | — | — |",
        "| Current-best FedPhoenixRG | **83.43%** | 984 | +0.82 pp | — | 81.27% | 979–989 | 81.84% | 83.39% |",
        result_row("FedPhoenixRG-SIR", runs["FedPhoenixRG-SIR"]),
        result_row("FedPhoenixRG-RMR", runs["FedPhoenixRG-RMR"]), "",
        "## Spectrum invariance", "",
        f"SIR maximum sorted-probability difference was {sir['max_spectrum_difference']:.3e}; "
        f"RMR was {rmr['max_spectrum_difference']:.3e}. Mean raw/new entropy was "
        f"{sir['raw_probability_entropy']:.6f}/{sir['new_probability_entropy']:.6f} "
        f"for SIR and {rmr['raw_probability_entropy']:.6f}/"
        f"{rmr['new_probability_entropy']:.6f} for RMR. The probability multiset, "
        "exploration floor, entropy, and concentration were preserved.", "",
        "## SIR ranking diagnostics", "",
        f"Mean Pearson/Spearman between raw q and q-SE ranking scores was "
        f"{sir['pearson_raw_q_vs_alternative']:.6f}/"
        f"{sir['spearman_raw_q_vs_alternative']:.6f}. Mean top-reset overlap was "
        f"{sir['top_reset_count_overlap']:.2%}; mean fraction of ranking positions "
        f"changed was {sir['fraction_ranking_positions_changed']:.2%}. Jackknife SE "
        f"mean/median-of-layer-medians/max was {sir['mean_jackknife_se']:.6f}/"
        f"{sir['median_jackknife_se']:.6f}/{sir['max_jackknife_se']:.6f}; mean "
        f"corr(SE,q) was {sir['corr_se_vs_q']:.6f}.", "",
        "## RMR ranking diagnostics", "",
        f"Mean Pearson/Spearman between q and positive R was "
        f"{rmr['pearson_q_vs_positive_r']:.6f}/"
        f"{rmr['spearman_q_vs_positive_r']:.6f}. Mean top-reset overlap was "
        f"{rmr['top_reset_count_overlap']:.2%}; mean fraction of ranking positions "
        f"changed was {rmr['fraction_ranking_positions_changed']:.2%}. Mean "
        f"corr(q,E) was {rmr['corr_q_vs_energy']:.6f}, and corr(positive R,E) was "
        f"{rmr['corr_positive_r_vs_energy']:.6f}.", "",
        f"Raw-q top filters had mean E/R+/q of {rmr['raw_top_mean_energy']:.6g}/"
        f"{rmr['raw_top_mean_positive_r']:.6g}/{rmr['raw_top_mean_q']:.6f}; "
        f"RMR top filters had {rmr['rmr_top_mean_energy']:.6g}/"
        f"{rmr['rmr_top_mean_positive_r']:.6g}/{rmr['rmr_top_mean_q']:.6f}.", "",
        "## Interpretation", "",
    ]
    if all(run["peak"] <= 83.43 for run in runs.values()):
        lines.append(
            "Under identical probability concentration, the original normalized "
            "client-balanced q ranking remains the strongest tested ordering signal. "
            "SIR closes stability reranking under this spectrum, and RMR closes "
            "absolute repeatable-magnitude reranking under this spectrum. No further "
            "q/ranking experiment was launched."
        )
    else:
        lines.append(
            "Any improvement between 83.46% and 83.50% is treated as marginal; only "
            "approximately 83.7% with neighborhood improvement is a clear positive "
            "signal. No follow-up was launched automatically."
        )
    lines += [
        "", "The original FedPhoenix and FedPhoenixRG function bodies and raw q "
        "finalizer remained unchanged. Exactly two separate 1000-round experiments "
        "were run, with no combination, sweep, third method, or follow-up.", "",
        "Pre-run validation: 32 focused tests passed. Both 20-round prefixes matched "
        "current-best RG exactly in selected clients, task seeds, reset assignments "
        "(same frozen builder and seeds), and round accuracy.",
    ]
    (RESULTS / "sir_rmr_1000_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    save_status(status="validating")
    if git("rev-parse", "HEAD") != STARTING_HEAD:
        raise RuntimeError("HEAD changed during formal runs")
    if git("branch", "--show-current") != "main":
        raise RuntimeError("branch changed during formal runs")
    if git("diff", "--cached", "--name-only"):
        raise RuntimeError("index contains unrelated staged files")
    runs = {method: load_run(method) for method in METHODS}
    verify_first_twenty(runs)
    assert_frozen_behavior()
    sir = summarize_sir(runs["FedPhoenixRG-SIR"]["diagnostics"])
    rmr = summarize_rmr(runs["FedPhoenixRG-RMR"]["diagnostics"])
    git("diff", "--check")
    write_summary(runs, sir, rmr)
    paths = [
        "Algorithm/repeatability_guidance.py", "main_fed.py",
        "train_five_baselines.py", "tests/test_spectrum_reranking.py",
        "results/run_sir_rmr_1000.py", "results/finalize_sir_rmr_1000.py",
        "results/sir_rmr_first20_validation/comparison.json",
        "results/sir_rmr_1000_summary.md", "results/rg_sir_1000",
        "results/rg_rmr_1000",
    ]
    for method, run in runs.items():
        paths.extend([run["metrics_path"], run["config_path"], run["diagnostics_path"]])
    git("add", "--", *paths)
    git("diff", "--cached", "--check")
    git("commit", "-m", "Test spectrum-preserving FedPhoenixRG reranking")
    sha = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    git("push", "origin", branch)
    save_status(
        status="pushed", starting_head=STARTING_HEAD,
        final_commit_sha=sha, branch=branch,
        results={
            method: {key: value for key, value in run.items() if key not in {
                "selected_clients", "task_seeds", "accuracies", "diagnostics"
            }} for method, run in runs.items()
        },
        sir_diagnostics=sir, rmr_diagnostics=rmr,
        summary="results/sir_rmr_1000_summary.md",
        exactly_two_formal_runs=True, combined=False, sweep=False,
        third_experiment=False, followup=False,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save_status(status="failed", error=repr(error))
        raise
