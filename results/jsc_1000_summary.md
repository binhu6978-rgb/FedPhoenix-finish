# FedPhoenixRG-JSC 1000-round experiment

Starting HEAD: `c7152f014264f3f6306d87e6705c63dbbd144280` on `main`.

## Result

| Method | Peak | Peak round | vs FedPhoenix | vs FedPhoenixRG | Round 1000 | Peak ±5 rounds | Neighborhood mean | Neighborhood second-best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedPhoenix | 82.61% | 997 | — | -0.82 pp | 80.59% | — | — | — |
| Current-best FedPhoenixRG | **83.43%** | 984 | +0.82 pp | — | 81.27% | 979–989 | 81.84% | 83.39% |
| FedPhoenixRG-JSC | 82.88% | 984 | +0.27 pp | -0.55 pp | 80.10% | 979–989 | 80.80% | 82.56% |

## Mechanism diagnostics

Statistics below are weighted by the active filter-rounds or active layer-rounds on which each refreshed distribution could affect training.

- Jackknife SE: mean 0.044356, median-of-layer/filter medians 0.029939, maximum 0.434272.
- Mean q-s: 0.037594; fraction s<q: 98.81%; fraction s=0: 8.40%.
- q versus s: Pearson 0.953135, Spearman 0.959230; top-reset-count overlap 79.72%.
- Eligible clients: mean 60.83, observed min 51, max 68.
- corr(SE, eligible clients): 0.006897; corr(SE, q): -0.372099; mean within-layer SE dispersion: 0.020859.
- Sampling entropy, raw q versus JSC: 5.913858 versus 5.861986; entropy gap from uniform: 0.026801 versus 0.078674.
- Mean max-probability/uniform ratio, raw q versus JSC: 1.519395 versus 1.875397.

## Interpretation

The fixed one-jackknife-SE calibration did not improve current-best RG.

JSC measures leave-one-eligible-client-out sensitivity of the unchanged client-balanced repeatability estimate. It is not interpreted as reset utility or as a frequentist confidence interval.

The original FedPhoenix and FedPhoenixRG function bodies, raw q estimator, schedule, perturbation, independent task sampling, aggregation, and local training remained unchanged. Exactly one new 1000-round formal experiment was run. No sweep or follow-up was launched.

Pre-run validation: 26 focused tests passed, and the 20-round validation matched current-best RG in selected clients, task seeds, and every accuracy.
