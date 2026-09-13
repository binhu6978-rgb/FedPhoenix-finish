# FedPhoenixRG q-remapping: 1000-round results

Peak Accuracy is the sole primary metric. Both runs use the existing
CIFAR-10/VGG16 fixed beta=0.3 partition, seed 1, 1000 rounds,
`rg_mix=0.75`, `rg_interval=20`, `FP_conv=1000`, reset ratio 1/64,
and `ori_normal`. Existing FedPhoenix and RG references were not rerun.

| Method | Peak | Peak Round | Delta vs FedPhoenix | Delta vs 83.43% |
|---|---:|---:|---:|---:|
| FedPhoenix | 82.61% | 997 | — | -0.82 pp |
| Current-best FedPhoenixRG | **83.43%** | 984 | +0.82 pp | — |
| FedPhoenixRG-Excess | 83.22% | 994 | +0.61 pp | -0.21 pp |
| FedPhoenixRG-Persistent | 83.23% | 987 | +0.62 pp | -0.20 pp |

Both experiments completed 1000 rounds with exit code 0. Each run
recorded 650 per-layer remapping diagnostics alongside the original
RG diagnostics. Complete logs, manifests, round metrics, and configs
are in `results/rg_remap_excess_1000/`,
`results/rg_remap_persistent_1000/`, and `results/training_metrics/`.

Neither remapping exceeded 83.43%. This q-remapping round stops here; no additional setting was tested.
