# FedPhoenixRG spectrum-preserving reranking experiments

Starting HEAD: `82b53c0a0f3c7b1d7f30d4442dcb36901a1f4ea8` on `main`.

## Results

| Method | Peak | Peak round | vs FedPhoenix | vs FedPhoenixRG | Round 1000 | Peak ±5 rounds | Neighborhood mean | Neighborhood second-best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedPhoenix | 82.61% | 997 | — | -0.82 pp | 80.59% | — | — | — |
| Current-best FedPhoenixRG | **83.43%** | 984 | +0.82 pp | — | 81.27% | 979–989 | 81.84% | 83.39% |
| FedPhoenixRG-SIR | 82.66% | 969 | +0.05 pp | -0.77 pp | 82.20% | 964–974 | 81.83% | 82.63% |
| FedPhoenixRG-RMR | 82.44% | 889 | -0.17 pp | -0.99 pp | 81.91% | 884–894 | 79.13% | 81.95% |

## Spectrum invariance

SIR maximum sorted-probability difference was 0.000e+00; RMR was 0.000e+00. Mean raw/new entropy was 5.909922/5.909922 for SIR and 5.910030/5.910030 for RMR. The probability multiset, exploration floor, entropy, and concentration were preserved.

## SIR ranking diagnostics

Mean Pearson/Spearman between raw q and q-SE ranking scores was 0.962269/0.964247. Mean top-reset overlap was 79.96%; mean fraction of ranking positions changed was 96.02%. Jackknife SE mean/median-of-layer-medians/max was 0.038290/0.034435/0.417647; mean corr(SE,q) was -0.321908.

## RMR ranking diagnostics

Mean Pearson/Spearman between q and positive R was 0.498658/0.505604. Mean top-reset overlap was 13.67%; mean fraction of ranking positions changed was 98.39%. Mean corr(q,E) was -0.179619, and corr(positive R,E) was 0.599684.

Raw-q top filters had mean E/R+/q of 0.000358725/7.38358e-05/0.264615; RMR top filters had 0.000680719/0.00010327/0.206141.

## Interpretation

Under identical probability concentration, the original normalized client-balanced q ranking remains the strongest tested ordering signal. SIR closes stability reranking under this spectrum, and RMR closes absolute repeatable-magnitude reranking under this spectrum. No further q/ranking experiment was launched.

The original FedPhoenix and FedPhoenixRG function bodies and raw q finalizer remained unchanged. Exactly two separate 1000-round experiments were run, with no combination, sweep, third method, or follow-up.

Pre-run validation: 32 focused tests passed. Both 20-round prefixes matched current-best RG exactly in selected clients, task seeds, reset assignments (same frozen builder and seeds), and round accuracy.
