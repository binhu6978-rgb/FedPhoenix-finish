# FedPhoenixRG 1200-round peak results

Both FedPhoenixRG runs used the current VGG/CIFAR-10 beta=0.3 partition,
seed 1, `rg_interval=20`, reset ratio 1/64, `ori_normal`, and `FP_conv=1000`.
Only `rg_mix` differed. They ran concurrently on GPU 0 and completed all
1200 rounds without a training error.

| Method | Peak accuracy | Peak round | Delta versus FedPhoenix |
|---|---:|---:|---:|
| FedPhoenix (existing baseline) | 83.04% | 1129 | — |
| FedPhoenixRG, mix=0.50 | 83.01% | 987 | -0.03 percentage points |
| FedPhoenixRG, mix=0.75 | **83.43%** | 984 | **+0.39 percentage points** |

Peak accuracy is the sole primary metric for this comparison. The existing
FedPhoenix baseline comes from
`result_other_method/cifar10/0.3/vgg_FedPhoenix.log`; it was not rerun.
These are single-seed results.

The complete FedPhoenixRG logs are `results/mix050_1200.stdout.log` and
`results/mix075_1200.stdout.log`. Their round-level metrics and configurations
are in `results/training_metrics/` with run names
`paired1200_rg_mix050` and `paired1200_rg_mix075`.
