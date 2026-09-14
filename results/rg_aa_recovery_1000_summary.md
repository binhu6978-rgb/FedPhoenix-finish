# FedPhoenixRG mechanism experiments: 1000-round results

Peak Accuracy is the sole primary selection metric. Both runs use
CIFAR-10/VGG16, the fixed beta=0.3 partition, seed 1, 1000 rounds,
`rg_mix=0.75`, `rg_interval=20`, `FP_conv=1000`, reset ratio 1/64,
and `ori_normal`. Existing FedPhoenix and RG references were not rerun.

| Method | Peak | Peak Round | vs FedPhoenix | vs 83.43% |
|---|---:|---:|---:|---:|
| FedPhoenix | 82.61% | 997 | — | -0.82 pp |
| Current-best FedPhoenixRG | **83.43%** | 984 | +0.82 pp | — |
| FedPhoenixRG-AA | 82.51% | 999 | -0.10 pp | -0.92 pp |
| FedPhoenixRG-Recovery | 82.39% | 987 | -0.22 pp | -1.04 pp |

## Aggregation-Aligned diagnostics

Across 650 layer-refresh rows, the mean absolute difference between equal-client q and AA q was 0.013485. Their mean Pearson correlation was 0.947982 (minimum 0.806282). Among 343 active-layer refreshes, mean top-reset overlap was 0.7198, and 87.5% changed at least one top-k filter. Eligible client sample counts ranged from 30 to 1447.

## Reset Recoverability diagnostics

Across 650 layer-refresh rows, mean rho was 0.622937; mean within-layer rho standard deviation was 0.248704, with observed range 0.000000–1.000000. The mean fraction of filters without a recovery observation was 62.3%, and the mean raw observation count per filter was 1.683. Mean correlation(q, rho) was 0.021210; mean correlation(q, u) was 0.549137 (minimum -0.168684). Fallback counts were {"recovery": 650}.

## Peak decision

Neither method exceeded 83.43%. The reset-placement optimization route stops here, and no additional variant was started.

Both runs completed 1000 rounds with exit code 0. Complete stdout,
stderr, manifests, round metrics, configs, and per-filter recovery
observation counts are saved under `results/`.
