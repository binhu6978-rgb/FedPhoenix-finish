# FedPhoenixRG 300-round mixture pilot

All runs use CIFAR-10, VGG16, the fixed beta=0.3 partition, 100 clients,
10% participation, 5 local epochs, seed 1, reset ratio 1/64,
`ori_normal`, and `FP_conv=1000`. Standard deviations are population standard
deviations over rounds 251-300.

| Method | Peak | Peak round | Last-50 mean | Trajectory mean | Rounds 21-300 mean | Round 300 | Last-50 std |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedPhoenix | 78.3500 | 218 | 70.7490 | 65.7027 | 68.1380 | 73.9300 | 8.3748 |
| FedPhoenixRG, mix=0.50 | 78.7100 | 284 | 71.9972 | 65.6693 | 68.1022 | 72.9400 | 4.6365 |
| FedPhoenixRG, mix=0.75 | 78.6800 | 290 | 71.3026 | 65.8116 | 68.2546 | 74.4100 | 6.3615 |

## Delta versus FedPhoenix

| Method | Peak | Last-50 mean | Trajectory mean | Rounds 21-300 mean | Round 300 | Last-50 std |
|---|---:|---:|---:|---:|---:|---:|
| mix=0.50 | +0.3600 | +1.2482 | -0.0334 | -0.0358 | -0.9900 | -3.7383 |
| mix=0.75 | +0.3300 | +0.5536 | +0.1088 | +0.1166 | +0.4800 | -2.0133 |

## Sampling diagnostics

Diagnostics below cover refreshes at rounds 20-280 (the round-300 scores were
not used by a later training round).

| Mix | Mean entropy gap | Max entropy gap | Mean max-probability / uniform | Max max-probability / uniform | Mean min-probability / uniform | Min min-probability / uniform |
|---|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.01027 | 0.09153 | 1.3011 | 2.4118 | 0.7422 | 0.5000 |
| 0.75 | 0.02314 | 0.19840 | 1.4354 | 3.5207 | 0.6115 | 0.2500 |

Both mixtures measurably depart from uniform sampling. Mix 0.75 creates the
stronger bias, while mix 0.50 gives the better Last-50 mean and substantially
lower Last-50 variability. For the next fixed configuration, mix 0.50 is the
more conservative recommendation. This single-seed pilot does not establish a
statistically general advantage.

## Commands

```powershell
python -u -X utf8 main_fed.py --algorithm FedPhoenixRG --dataset cifar10 --model vgg --epochs 300 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 50 --bs 256 --optimizer sgd --lr 0.01 --momentum 0.5 --weight_decay 0 --iid 0 --noniid_case 5 --data_beta 0.3 --FP_conv 1000 --FP_fc 0 --reset 0.015625 --remethod ori_normal --num_classes 10 --generate_data 0 --seed 1 --eval_every 1 --gpu 0 --rg_interval 20 --rg_mix 0.50 --run_name paired300_rg_mix050

python -u -X utf8 main_fed.py --algorithm FedPhoenixRG --dataset cifar10 --model vgg --epochs 300 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 50 --bs 256 --optimizer sgd --lr 0.01 --momentum 0.5 --weight_decay 0 --iid 0 --noniid_case 5 --data_beta 0.3 --FP_conv 1000 --FP_fc 0 --reset 0.015625 --remethod ori_normal --num_classes 10 --generate_data 0 --seed 1 --eval_every 1 --gpu 0 --rg_interval 20 --rg_mix 0.75 --run_name paired300_rg_mix075
```
