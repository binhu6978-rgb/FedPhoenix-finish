# FedPhoenixRG 1000-round follow-up

Peak Accuracy is the sole primary metric. All runs use CIFAR-10, VGG16, the
fixed beta=0.3 partition, seed 1, 100 clients, 10% participation, five local
epochs, reset ratio 1/64, `ori_normal`, and `rg_interval=20`.

| Method | rg_mix | FP_conv | Peak Accuracy | Peak Round | Delta vs FedPhoenix | Delta vs current best |
|---|---:|---:|---:|---:|---:|---:|
| FedPhoenix | — | 1000 | 82.61% | 997 | — | -0.82 pp |
| Current best FedPhoenixRG | 0.75 | 1000 | **83.43%** | 984 | +0.82 pp | — |
| Experiment A | 0.75 | 900 | 82.97% | 1000 | +0.36 pp | -0.46 pp |
| Experiment B | 0.90 | 1000 | 82.30% | 850 | -0.31 pp | -1.13 pp |

The existing `rg_mix=0.75, FP_conv=1000` configuration remains best. Ending
the reset schedule at round 900 did not improve Peak Accuracy, and increasing
the mixture coefficient to 0.90 reduced it. No further setting was tested.

The FedPhoenix and current-best references are computed from rounds 1-1000 of
their existing 1200-round logs. Neither reference was rerun. Both new runs
completed 1000 rounds without a training error.
