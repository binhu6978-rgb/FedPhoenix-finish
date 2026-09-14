# FedPhoenixRG remaining-mechanism assessment

Starting HEAD: `3529931f35cd50f37fdd824832e99a1d6f73365b` on `main`.

## Reconstructed intervention cycle

Each round samples 10 clients, creates 10 independent CPU task copies of the
same global model, chooses a fixed number of filters in every active
convolutional layer, and replaces those filters with task-private `ori_normal`
draws. Each selected client trains its assigned task copy for five local epochs.
The server then sample-count averages the absolute post-local model states.
FedPhoenixRG changes only filter selection: every 20 rounds it estimates
client-balanced repeatability `q` from non-reset observations and uses a
25% uniform / 75% normalized-q mixture from the next round onward.

The stair schedule, perturbation values, local optimizer, and aggregation do
not use `q`. Aggregation does not know that reset clients started selected
filters from a different state.

## Evidence from completed runs

For current-best RG, active-layer diagnostics have mean `q=0.175`, 98.9% of
filters positive, and a mean entropy gap from uniform of only 0.0268 nats.
Alternative placement rules changed rankings materially but did not improve
Peak: LC 82.64%, Excess 83.22%, Persistent 83.23%, AA 82.51%, and q×recovery
82.39%. AA changed at least one top-reset filter in 87.5% of active refreshes.
This makes further placement remapping a low-priority direction.

The Recovery run provides evidence about the intervention itself. Restricting
its logs to active layers, 30.6% of filters lacked an observation; among
observed filters, mean recovery toward the same-round non-reset post-local
consensus was approximately 0.0009. Full random replacement therefore remains
far from the non-reset state after five local epochs. This does not show that
consensus recovery is desirable, but it shows that the reset start remains a
large component of the endpoint consumed by aggregation.

## Candidate ranking

Scores range from 1 (weak) to 5 (strong).

| Rank | Mechanism question | Evidence | Orthogonality | Peak upside | Interpretability | Simplicity | Fair compute | Negative-result value | Risk |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Should aggregation remove the raw reset-start offset and aggregate each client's learned delta from its actual task start? | 5 | 5 | 4 | 5 | 4 | 5 | 5 | Medium |
| 2 | Can a task-private within-filter permutation preserve filter scale/information while still disrupting feature semantics enough to drive relearning? | 4 | 5 | 4 | 5 | 4 | 5 | 5 | Medium |
| 3 | Should the first local epoch update reset filters only before returning to ordinary local SGD? | 3 | 5 | 3 | 4 | 3 | 4 | 4 | High |
| 4 | Should intervention timing respond to online training state rather than the fixed stair schedule? | 2 | 4 | 3 | 3 | 2 | 5 | 3 | High |

## Formal selections

### FedPhoenixRG-DeltaAgg

For a reset client/filter, ordinary aggregation consumes the absolute endpoint
`theta_local = theta_reset + delta_local`. The variant contributes
`theta_global + delta_local` instead, using the same client sample weight. It
therefore removes only the raw reset initialization offset while preserving
the local gradient path induced by that reset. Non-reset parameters are
unchanged. This is a single aggregation-semantics change with no new
hyperparameter and no singleton amplification.

### FedPhoenixRG-Permute

The selected filter indices remain exactly those of current-best RG. Instead
of drawing entirely new values from layer-wide `ori_normal` statistics, each
selected filter receives a task-private random permutation of its own scalar
weights. This preserves that filter's exact mean, variance, and norm while
destroying its channel/spatial arrangement. It tests whether the remaining
bottleneck is information destruction by the reset operator. Standard local
training and ordinary absolute-state aggregation remain unchanged.

The candidates are independent and will not be combined. The reset-only local
warmup was rejected because it removes one fifth of ordinary parameter updates
and confounds recovery specialization with reduced base training. Adaptive
timing was rejected because existing `q` and recovery logs do not establish
that any reset is beneficial or harmful at a particular time; implementing it
would require an additional utility estimator or threshold.
