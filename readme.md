Use AutoRun.py to run the FedPhoenix algorithm and the Baselines algorithm

## Direct default run

The command-line defaults are configured for the current primary run:
Recovery-Aware FedPhoenix on CIFAR-10 with ResNet-18, Dirichlet beta 0.3,
100 clients, 10% participation, seed 1, and 1200 communication rounds.  The
recovery configuration uses assignment-only matching, starts at round 50, and
applies a 0.6 confidence gate.  In the `fd` environment, run:

```powershell
python main_fed.py
```

Command-line flags can still override every default.

## Run the six original baselines

To sequentially run the original repository implementations of FedAvg,
FedProx, FedMut, ClusteredSampling (CluSample), FedGen, and FedPhoenix with the same shared
CIFAR-10/ResNet-18/beta-0.3/seed-1 settings, use:

```powershell
python train_six_baselines.py
```

The launcher does not replace the methods' original training loops.  These
six original baselines evaluate every round on the complete 10,000-sample
CIFAR-10 server test set.  The launcher saves separate stdout/stderr logs and a JSON manifest under
`results/six_baselines_logs/<timestamp>/`.  Use `--dry-run` to print all six
commands without starting training.

All comparison methods, including FedPhoenix and FedPhoenixRecovery, evaluate
every communication round on the complete server test set and print exactly one
machine-readable accuracy record, for example
`ROUND_ACCURACY method=FedAvg round=1 accuracy=25.000000`.  At the end it
prints `PEAK_ACCURACY` with the highest accuracy and its round.  The baseline
loops do not print test loss or save redundant model/result files.

Sure, I can help you with that. Here's an MD file that explains the usage of the provided script:
# Experiment Configuration and Execution

## Introduction
This script is used to configure and run various experiments for federated learning algorithms. It allows you to specify different datasets, models, algorithms, and various training-related parameters.

## Usage
1. **Dataset Related Parameters**:
   - The `datasets` list specifies the datasets to be used for the experiments.
   - The `dataset_classes` dictionary defines the number of classes for each dataset.

2. **Algorithm Related Parameters**:
   - The `algorithms` list specifies the federated learning algorithms to be used.

3. **Model Related Parameters**:
   - The `models` list specifies the models to be used for the experiments.

4. **Training Related Parameters**:
   - `lrs`: Learning rates to be used.
   - `epochs`: Number of training epochs.
   - `data_betas`: Data distribution parameters, where 0.5 indicates IID data and other values indicate non-IID data.
   - `weight_decays`: Weight decay values.

5. **Federated Learning Related Parameters**:
   - `num_users`: Total number of clients.
   - `frac`: Fraction of clients participating in each round.

6. **Specific Algorithm Parameters**:
   - `fp_convs`: The total number of rounds for resetting in the Convolutional Layer of the FedPhoenix algorithm.Corresponding to r_s in the paper
   - `resets`: FedPhoenix algorithm's reset rate parameter.Corresponding to θ in the paper

7. **Hardware Related Parameters**:
   - `gpus`: List of GPU IDs to be used for the experiments.

8. **Resource Management Parameters**:
   - `GPU_MEMORY_THRESHOLD`: GPU memory usage threshold.
   - `GPU_UTIL_THRESHOLD`: GPU utilization threshold.
   - `TASK_INIT_SLEEP`: Task initialization wait time.
   - `SCHEDULER_SLEEP_BUSY`: Scheduler busy polling interval.
   - `SCHEDULER_SLEEP_IDLE`: Scheduler idle polling interval.

## Parameter Validation
The script includes a `validate_params()` function that checks the validity of the parameter configuration, ensuring that the values are within the expected ranges.

## Parameter Combinations
The script uses `itertools.product()` to generate all possible combinations of the specified parameters. It also dynamically generates the `fixed_params` dictionary based on the `data_beta` parameter, indicating whether the data is IID or non-IID.

## GPU Manager
The script includes a `GPUManager` class that is responsible for monitoring and managing the status and task allocation of multiple GPUs. It provides methods to check GPU availability, get the best available GPU, and add/clean up running processes.

## Main Function
The `main()` function is the entry point of the script. It initializes the `GPUManager`, schedules tasks on available GPUs, and manages the running tasks.

## Reset-Recovery Motivation Experiment

`motivation_experiment.py` tests whether client value depends on the concrete
FedPhoenix reset task. It keeps the global checkpoint and probe-client set
fixed, replays deterministic reset copies on every probe client, and reports:

1. whether the same reset has different outcomes across clients;
2. whether client rankings change across reset tasks after removing general
   client utility and task difficulty effects;
3. whether changing only the client--reset assignment changes the aggregated
   model.

Recommended CIFAR-10 pilot:

```powershell
python motivation_experiment.py `
  --dataset cifar10 --model cnn `
  --num-users 20 --dirichlet-beta 0.1 `
  --warmup-rounds 100 --warmup-clients 5 `
  --checkpoint-selection-samples 1000 --eval-samples 5000 `
  --probe-clients 4 --probe-selection coverage `
  --num-tasks 4 --repeats 3 --exact-assignment-limit 720 `
  --bootstrap-samples 2000 --confidence-level 0.95 `
  --reset-ratio 0.25 `
  --output-dir results/motivation_cifar10
```

Checkpoint selection and final hypothesis evaluation use disjoint holdout
samples. The experiment writes `summary.json`, `REPORT.md`, detailed raw CSV
tables, a warm-up checkpoint, and `motivation_summary.png`. Plot-ready files
are `figure_recovery_matrix.csv`, `figure_assignment_losses.csv`, and
`figure_key_metrics.csv`. When the assignment space is small enough, every
permutation is evaluated rather than sampling random assignments. The
score-matched assignment is an oracle motivation diagnostic computed from the
completed recovery matrix; it is not a deployable selector. A single global
seed supports a controlled within-run mechanism claim, not cross-seed
robustness.

To regenerate the report and figure without retraining:

```powershell
python motivation_experiment.py --postprocess-only `
  --output-dir results/motivation_cifar10
```

For an offline pipeline smoke test without downloading CIFAR-10, use
`--dataset synthetic`. Synthetic results validate code execution only and must
not be used as research evidence.

## Recovery-Aware FedPhoenix Baseline

`FedPhoenixRecovery` is the deployable non-Agent baseline for the proposed
method. In each round it:

1. creates the reset copies before choosing clients;
2. samples a candidate pool larger than the participation budget;
3. uses a small private support/query probe to estimate one-step recovery for
   every client--copy pair;
4. constructs a standardized recovery score from functional recovery gain,
   advantage over the unreset model, residual damage, and reset-gradient
   alignment;
5. applies maximum-weight one-to-one matching and trains each selected client
   from its assigned copy.

Example:

```powershell
python main_fed.py `
  --algorithm FedPhoenixRecovery `
  --dataset cifar10 --model cnn `
  --num_users 20 --frac 0.1 `
  --generate_data 1 --iid 0 --noniid_case 5 --data_beta 0.1 `
  --reset 0.25 --FP_conv 100 `
  --seed 1 --eval_every 5 `
  --metrics_log_dir results/training_metrics `
  --run_name joint `
  --recovery_mode joint `
  --recovery_pool_multiplier 3 `
  --recovery_probe_samples 64
```

Per-pair features, scores, selected assignments, and matching time are written
to `--recovery_log_dir`. Unified per-round accuracy, loss, timing, selected
clients, assignments, and reset-task seeds are written to
`--metrics_log_dir`. Use `--recovery_mode assignment_only` to keep random
clients and optimize only copy assignment, or `--recovery_mode selection_only`
to select clients by mean recovery score and randomize copy assignment. Use
`--recovery_mode interaction_joint` to remove general client and task main
effects before joint matching, retaining only pair-specific recovery value.
The future shared-policy Agent should replace the score generator while
retaining the same task-generation and constrained-matching interfaces.
