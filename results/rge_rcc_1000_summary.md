# FedPhoenixRG RGE/RCC experiments

Starting HEAD: `4c659969d28f35f302d07452cb315094fccbbcf5` on `main`.

## Results

| Method | Peak | Peak round | vs FedPhoenix | vs FedPhoenixRG | Round 1000 | Peak ±5 rounds | Neighborhood mean | Neighborhood second-best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedPhoenix | 82.61% | 997 | — | -0.82 pp | 80.59% | — | — | — |
| Current-best FedPhoenixRG | **83.43%** | 984 | +0.82 pp | — | 81.27% | 979–989 | 81.84% | 83.39% |
| FedPhoenixRG-RGE | 82.77% | 934 | +0.16 pp | -0.66 pp | 78.59% | 929–939 | 81.79% | 82.35% |
| FedPhoenixRG-RCC | 82.25% | 848 | -0.36 pp | -1.18 pp | 76.15% | 843–853 | 77.91% | 81.18% |

## RGE mechanism diagnostics

The gate triggered for 12 of 13 layer boundaries. Details:

| Layer | Original stop round | Latest q | Earlier mean q | Retention | Triggered | Logged extension rounds | Extension mean q | Extension mean entropy | Extension mean accuracy |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `features.0` | 78 | 0.129396 | 0.130185 | 0.993934 | yes | 20 | 0.147972 | 4.089488 | 67.845000 |
| `features.3` | 155 | 0.193160 | 0.146483 | 1.318653 | yes | 20 | 0.183580 | 4.153785 | 68.803000 |
| `features.7` | 232 | 0.180933 | 0.162221 | 1.115353 | yes | 20 | 0.178719 | 4.848224 | 71.398000 |
| `features.10` | 309 | 0.166961 | 0.159376 | 1.047592 | yes | 20 | 0.158605 | 4.849638 | 72.565001 |
| `features.14` | 386 | 0.145635 | 0.162261 | 0.897537 | yes | 20 | 0.142885 | 5.541672 | 72.889500 |
| `features.17` | 463 | 0.155689 | 0.151725 | 1.026122 | yes | 20 | 0.150800 | 5.541202 | 74.299000 |
| `features.20` | 540 | 0.101926 | 0.147366 | 0.691650 | yes | 20 | 0.099356 | 5.540245 | 73.025999 |
| `features.24` | 617 | 0.129573 | 0.154594 | 0.838148 | yes | 20 | 0.113027 | 6.228811 | 72.588000 |
| `features.27` | 694 | 0.155079 | 0.145227 | 1.067838 | yes | 20 | 0.137341 | 6.228728 | 78.166500 |
| `features.30` | 771 | 0.082292 | 0.126551 | 0.650267 | yes | 20 | 0.086201 | 6.203894 | 75.769500 |
| `features.34` | 848 | 0.187722 | 0.249548 | 0.752250 | yes | 20 | 0.166881 | 6.183096 | 75.279000 |
| `features.37` | 925 | 0.271656 | 0.234448 | 1.158706 | yes | 20 | 0.255075 | 6.226801 | 80.284000 |
| `features.40` | 1001 | 0.064331 | 0.166769 | 0.385752 | no | 0 | — | — | — |

## RCC mechanism diagnostics

Across 6746 guided layer-rounds, coordinated sampling used 443330 reset slots and produced 0 duplicate slots (100.00% unique). The private independent counterfactual produced 31787 duplicate slots (92.83% unique).

Mean selected q was 0.188720 for RCC and 0.189553 for the independent counterfactual. Mean selected probability was 0.002487 versus 0.002495; maximum multiplicity was 1 versus 4.

## Interpretation

Neither independent mechanism reached the approximately 83.7% threshold for a clear single-seed improvement with neighborhood support. Under the fixed framework, current FedPhoenixRG is likely near a local performance ceiling. No further seed=1 method search was launched.

Original `FedPhoenix` and `FedPhoenixRG` function bodies and the q estimator remained unchanged. Exactly two independent runs were executed; there was no combined or third variant.

Focused validation: 27 tests passed before formal launch.
