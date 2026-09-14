# FedPhoenixRG remaining-mechanism experiments

Starting HEAD: `3529931f35cd50f37fdd824832e99a1d6f73365b` on `main`.

## Research selection

Placement was deprioritized because current-best RG already gives
positive q to 98.9% of active filters with only a 0.0268-nat mean
entropy gap, while LC, Excess, Persistent, AA, and q×recovery all
failed to improve Peak. Four candidate mechanism questions were scored:

| Rank | Candidate | Evidence | Orthogonality | Upside | Interpretability | Simplicity | Fair compute | Negative value | Risk |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Actual-task delta aggregation | 5 | 5 | 4 | 5 | 4 | 5 | 5 | Medium |
| 2 | Within-filter permutation | 4 | 5 | 4 | 5 | 4 | 5 | 5 | Medium |
| 3 | One-epoch reset-only warmup | 3 | 5 | 3 | 4 | 3 | 4 | 4 | High |
| 4 | State-aware intervention timing | 2 | 4 | 3 | 3 | 2 | 5 | 3 | High |

The first two were selected because they test independent aggregation
and perturbation bottlenecks with one discrete change each. Warmup was
rejected because it removes one fifth of ordinary parameter updates;
timing was rejected because existing diagnostics provide no direct
per-round intervention utility signal.

## Results

| Method | Peak | Peak round | vs FedPhoenix | vs FedPhoenixRG | Round 1000 | Peak-neighborhood mean | Neighborhood second-best |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedPhoenix | 82.61% | 997 | — | -0.82 pp | 80.59% | — | — |
| Current-best FedPhoenixRG | **83.43%** | 984 | +0.82 pp | — | 81.27% | 81.84% | 83.39% |
| FedPhoenixRG-DeltaAgg | 77.84% | 600 | -4.77 pp | -5.59 pp | 77.08% | 76.60% | 77.57% |
| FedPhoenixRG-Permute | 82.43% | 934 | -0.18 pp | -1.00 pp | 73.08% | 79.83% | 82.09% |

Peak neighborhoods are clipped to rounds 1–1000 and otherwise cover
peak-5 through peak+5.

## FedPhoenixRG-DeltaAgg

Scientific question: does ordinary absolute-state averaging confuse
the random reset start with a learned client update? Reset filters
contribute `theta_global + (theta_local - theta_reset)` at their original
sample weight; every other parameter uses ordinary FedAvg.

Across 7006 layer-round diagnostics and 456530 reset observations, mean perturbation norm was 0.840407, mean local-step norm was 0.013084, and their mean ratio was 0.015497. Mean perturbation/local-step cosine was -0.003736; mean aggregate correction norm was 0.743500. The mean fraction of reset filters hit by multiple clients in a round was 7.9%.

FedPhoenixRG-DeltaAgg did not exceed 83.43%; its mechanism hypothesis is weakened.

## FedPhoenixRG-Permute

Scientific question: does full layer-distribution resampling destroy
too much filter-specific information? Selected indices use the same RG
sampler, but task-private permutations preserve each selected filter's
exact values, mean, variance, and norm while disrupting arrangement.

Across 7006 layer-round diagnostics and 456530 reset observations, mean relative perturbation norm was 1.366764, mean original/permuted cosine was 0.060654, and the mean fraction of scalar positions left fixed was 0.000375.

FedPhoenixRG-Permute did not exceed 83.43%; its mechanism hypothesis is weakened.

## Overall interpretation

Both independent mechanism tests failed to exceed current-best RG. Together with the prior placement negatives, this supports the conclusion that FedPhoenixRG is likely near a local performance ceiling under the current fixed framework. No follow-up was started.

Original FedPhoenixRG behavior remained unchanged. No hyperparameter
sweep, combined variant, or unplanned follow-up method was run.

Complete stdout/stderr, commands, manifests, metrics, configs, and
mechanism diagnostics are stored in `results/rg_deltaagg_1000/`,
`results/rg_permute_1000/`, and `results/training_metrics/`.
