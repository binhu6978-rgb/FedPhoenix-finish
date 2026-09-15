# RGE/RCC pre-run evidence and fixed designs

Starting HEAD: `4c659969d28f35f302d07452cb315094fccbbcf5` on `main`.

## RGE evidence

The current-best `results/mix075_1200.stdout.log` was parsed directly. For
each layer, the last complete RG refresh before its original stair boundary
was divided by the mean of its earlier active refreshes.

| Layer | Original boundary | Last refresh | Latest mean q | Earlier-window mean q | Retention |
|---|---:|---:|---:|---:|---:|
| `features.34` | 846.153846 | 840 | 0.160575 | 0.250214 | **0.641750** |
| `features.37` | 923.076923 | 920 | 0.152838 | 0.237528 | **0.643453** |
| `features.40` | 1000.000000 | 1000 | 0.180916 | 0.165366 | **1.094032** |

Thus `features.34` and `features.37` both meet the fixed 0.5 threshold at
their in-budget stop boundaries. `features.40` reaches its boundary only
after round 1000, so its decision is diagnostic and cannot change this run.
RGE keeps the original schedule, adds exactly rounds 848–867 for
`features.34` and rounds 925–944 for `features.37` if the online run makes
the same gate decisions, and never rechecks a layer.

## RCC evidence and design

With ten tasks and reset ratio 1/64, independent uniform task sampling has
an expected duplicate-slot fraction of about 7.5% for 64-filter layers and
6.6%–6.7% for 128–512-filter layers. The prior DeltaAgg trace diagnostic,
which used the same task count and reset sampler, measured 7.9% of affected
filters being hit by multiple clients on average. The exact current-best
reset indices were not stored, so no unsupported reconstruction is claimed.

RCC is disabled while guidance is absent or a layer's weights are constant,
which preserves the original RG path through round 20 and exact-uniform
fallbacks. Otherwise it draws all task slots for a layer as one exact
Efraimidis–Spirakis weighted sample without replacement from the unchanged
current-best probability vector, privately shuffles the selected filters,
and partitions them equally across tasks. In this fixed protocol, total
slots are below the layer width, so this gives unique round-level coverage
without changing any task's reset count. Reset values retain the original
task-private torch seeds. An independent counterfactual is replayed from the
original task seeds only for diagnostics and never touches training RNG.

RGE and RCC are separate branches. No combined or third variant is planned.
