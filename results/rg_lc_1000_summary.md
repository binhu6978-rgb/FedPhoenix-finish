# FedPhoenixRG-LC: 1000-round result

Peak Accuracy is the sole primary metric. This single run used the existing
CIFAR-10/VGG16 beta=0.3 partition, seed 1, `rg_interval=20`, base
`rg_mix=0.75`, `FP_conv=1000`, reset ratio 1/64, and `ori_normal`. All other
training settings match the current-best FedPhoenixRG run.

| Method | Peak Accuracy | Peak Round | Delta vs FedPhoenix | Delta vs 83.43% |
|---|---:|---:|---:|---:|
| FedPhoenix (existing) | 82.61% | 997 | — | -0.82 pp |
| FedPhoenixRG, mix=0.75 (existing best) | **83.43%** | 984 | +0.82 pp | — |
| FedPhoenixRG-LC | 82.64% | 987 | +0.03 pp | **-0.79 pp** |

LC completed all 1000 rounds with exit code 0. The prior reference runs were
not rerun. The first 20 LC accuracies exactly match the current-best RG
metrics, before calibrated sampling starts at round 21. The FedPhoenix and RG
functions remain unchanged in the current source tree.

The LC guidance refresh logged 343 active-layer observations over 49 refresh
rounds (20 through 980). `gamma_l` ranged from 0.286 to 0.750, with mean
0.652 and median 0.678; 122/343 observations reached the 0.75 ceiling.

| Refresh rounds | Active-layer observations | Mean gamma_l | Range |
|---|---:|---:|---:|
| 20–200 | 120 | 0.669 | 0.429–0.750 |
| 220–400 | 95 | 0.649 | 0.446–0.750 |
| 420–600 | 69 | 0.622 | 0.412–0.750 |
| 620–800 | 42 | 0.642 | 0.286–0.750 |
| 820–980 | 17 | 0.685 | 0.518–0.750 |

Layer behavior was mixed rather than uniformly increasing with depth.
`features.30` had the lowest mean `gamma_l` (0.520; 38 refreshes), while
`features.34` and `features.37` were near the ceiling (0.749 and 0.748;
42 and 46 refreshes). The final `features.40` layer averaged 0.651 across
49 refreshes and reached 0.750 once it became the only active layer: at that
point its `S_l` equals `S_bar` by definition. This supports attenuation of
weak-repeatability layers, but does not establish that every deeper layer
receives stronger guidance.

Sampling entropy averaged 0.020 nats below the corresponding uniform entropy
across the logged active-layer observations. Mean eligible clients per filter
averaged 60.82; the smallest logged minimum was 51. Eligible-client counts
were diagnostic only.

The run had no training error, but the late trajectory remained volatile:
round 1000 accuracy was 73.30%, and 12 of rounds 901–1000 fell below 75%.
The existing RG run also had 13 such rounds, so this is not unique evidence
of an LC-specific failure. Under the specified Peak-only criterion, LC did
not improve upon the current-best RG configuration. No other configuration
was tested.

Complete stdout/stderr and the run manifest are in
`results/rg_lc_mix075_fp1000_1000/20260913_164351/`. Round-level metrics and
configuration are in `results/training_metrics/` under the
`cifar10_vgg_FedPhoenixRG-LC_seed1_original_FedPhoenixRG-LC_seed1_1000` stem.
