"""Timing and round-level coverage variants built on FedPhoenixRG."""

from __future__ import annotations

from collections import Counter
import copy
import math
import random

import torch

from Algorithm.Phoenix_util import _sample_kernel_indices, reset_kernels_for_task


def original_layer_schedule(layers, conv_transition_period):
    """Return original stair boundaries without changing FedPhoenix semantics."""
    total_layers = len(layers)
    return {
        name: {
            "depth": depth,
            "boundary": (depth + 1) * float(conv_transition_period) / total_layers,
            "first_inactive_iter": int(
                math.ceil((depth + 1) * float(conv_transition_period) / total_layers)
            ),
        }
        for depth, (name, _state_key, _filters) in enumerate(layers)
    }


class OneWindowExtensionController:
    """Give each layer at most one fixed RG-window schedule extension."""

    def __init__(
        self,
        layers,
        conv_transition_period,
        rg_interval,
        threshold=0.5,
        epsilon=1e-12,
    ):
        if int(rg_interval) < 1:
            raise ValueError("rg_interval must be positive")
        self.layers = list(layers)
        self.schedule = original_layer_schedule(
            self.layers, conv_transition_period
        )
        self.interval = int(rg_interval)
        self.threshold = float(threshold)
        self.epsilon = float(epsilon)
        self.history = {name: [] for name, _key, _filters in self.layers}
        self.decisions = {}

    def record_refresh(self, round_number, scores, sampling_weights):
        """Record refreshes that occurred before each original stop boundary."""
        for name, _key, _filters in self.layers:
            original_last_active_round = self.schedule[name]["first_inactive_iter"]
            if int(round_number) > original_last_active_round:
                continue
            score = torch.as_tensor(scores[name], dtype=torch.float64)
            probability = torch.as_tensor(
                sampling_weights[name], dtype=torch.float64
            )
            entropy = float(
                -(probability * probability.clamp_min(1e-300).log()).sum()
            )
            self.history[name].append(
                {
                    "round": int(round_number),
                    "mean_q": float(score.mean()),
                    "sampling_entropy": entropy,
                }
            )

    def _make_decision(self, name):
        values = self.history[name]
        latest = values[-1]["mean_q"] if values else 0.0
        previous = [row["mean_q"] for row in values[:-1]]
        previous_mean = sum(previous) / len(previous) if previous else 0.0
        retention = (
            latest / (previous_mean + self.epsilon)
            if previous_mean > self.epsilon
            else 0.0
        )
        gate = bool(previous and retention >= self.threshold)
        first_iter = self.schedule[name]["first_inactive_iter"]
        decision = {
            "event": "rge_gate",
            "layer_name": name,
            "original_stop_round": first_iter + 1,
            "original_boundary": self.schedule[name]["boundary"],
            "mean_q_history": list(values),
            "latest_q_at_stop": latest,
            "previous_active_mean_q": previous_mean,
            "retention_ratio": retention,
            "retention_threshold": self.threshold,
            "gate_triggered": gate,
            "extended_rounds": self.interval if gate else 0,
            "extension_start_round": first_iter + 1 if gate else None,
            "extension_end_round": first_iter + self.interval if gate else None,
        }
        self.decisions[name] = decision
        return decision

    def forced_active_layers(self, current_iter):
        """Return the layers inside their single extension at this iteration."""
        active = set()
        new_decisions = []
        for name in self.schedule:
            first_iter = self.schedule[name]["first_inactive_iter"]
            if current_iter < first_iter:
                continue
            if name not in self.decisions:
                new_decisions.append(self._make_decision(name))
            decision = self.decisions[name]
            if (
                decision["gate_triggered"]
                and current_iter < first_iter + self.interval
            ):
                active.add(name)
        return active, new_decisions


def build_rge_fedphoenix_tasks(
    net_glob,
    round_idx,
    task_count,
    args,
    sampling_weights,
    forced_active_layers,
):
    """Build ordinary RG tasks with only selected layer stop times extended."""
    task_models = []
    task_traces = []
    for task_id in range(int(task_count)):
        task_model = copy.deepcopy(net_glob).to("cpu")
        task_seed = (
            int(args.seed) + 200000 + int(round_idx) * 100003
            + int(task_id) * 997
        )
        trace = reset_kernels_for_task(
            task_model,
            reset_ratio=args.reset,
            seed=task_seed,
            layer_scope="all",
            init_method=args.remethod,
            at_least_one=False,
            current_iter=round_idx,
            conv_transition_period=args.FP_conv,
            sampling_weights=sampling_weights,
            forced_active_layers=forced_active_layers,
        )
        task_models.append(task_model)
        task_traces.append(trace)
    return task_models, task_traces


def _valid_nonconstant_weights(weights, num_kernels):
    try:
        values = [
            float(item)
            for item in torch.as_tensor(weights).flatten().tolist()
        ]
    except (TypeError, ValueError, RuntimeError):
        return None
    if (
        len(values) != int(num_kernels)
        or any(not math.isfinite(value) or value < 0.0 for value in values)
        or sum(values) <= 0.0
        or max(values) == min(values)
    ):
        return None
    return values


def _weighted_round_assignment(values, reset_per_task, task_count, seed):
    """Exact weighted sampling without replacement across round-level slots."""
    num_kernels = len(values)
    total_slots = int(reset_per_task) * int(task_count)
    if total_slots > num_kernels:
        return None
    rng = random.Random(int(seed))
    keys = [
        (-math.log(max(rng.random(), 1e-300)) / weight, index)
        for index, weight in enumerate(values)
        if weight > 0.0
    ]
    if len(keys) < total_slots:
        return None
    keys.sort()
    selected = [index for _key, index in keys[:total_slots]]
    rng.shuffle(selected)
    return {
        task_id: sorted(
            selected[
                task_id * reset_per_task:(task_id + 1) * reset_per_task
            ]
        )
        for task_id in range(int(task_count))
    }


def build_coordinated_fedphoenix_tasks(
    net_glob, round_idx, task_count, args, sampling_weights=None
):
    """Coordinate guided reset slots while preserving every task's budget."""
    conv_layers = [
        (name, module) for name, module in net_glob.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    task_count = int(task_count)
    assignments = {task_id: {} for task_id in range(task_count)}
    total_layers = len(conv_layers)
    for depth, (name, layer) in enumerate(conv_layers):
        if round_idx >= (depth + 1) * float(args.FP_conv) / total_layers:
            continue
        num_kernels = int(layer.weight.shape[0])
        reset_per_task = min(
            num_kernels, int(num_kernels * float(args.reset))
        )
        if reset_per_task <= 0 or sampling_weights is None:
            continue
        values = _valid_nonconstant_weights(
            sampling_weights.get(name), num_kernels
        )
        if values is None:
            continue
        layer_assignment = _weighted_round_assignment(
            values,
            reset_per_task,
            task_count,
            seed=(
                int(args.seed) + 900000 + int(round_idx) * 100003
                + depth * 7919
            ),
        )
        if layer_assignment is None:
            continue
        for task_id, indices in layer_assignment.items():
            assignments[task_id][name] = indices

    task_models = []
    task_traces = []
    for task_id in range(task_count):
        task_model = copy.deepcopy(net_glob).to("cpu")
        task_seed = (
            int(args.seed) + 200000 + int(round_idx) * 100003
            + task_id * 997
        )
        trace = reset_kernels_for_task(
            task_model,
            reset_ratio=args.reset,
            seed=task_seed,
            layer_scope="all",
            init_method=args.remethod,
            at_least_one=False,
            current_iter=round_idx,
            conv_transition_period=args.FP_conv,
            sampling_weights=sampling_weights,
            reset_indices_by_layer=assignments[task_id],
        )
        task_models.append(task_model)
        task_traces.append(trace)
    return task_models, task_traces


def coverage_diagnostics(
    task_traces, sampling_weights=None, scores=None, counterfactual=False
):
    """Summarize round/layer multiplicity and selected q/probabilities."""
    layer_indices = {}
    layer_sizes = {}
    for trace in task_traces:
        for row in trace.get("layers", []):
            name = row["name"]
            layer_sizes[name] = int(row["num_kernels"])
            layer_indices.setdefault(name, []).extend(
                int(index) for index in row["reset_indices"]
            )
    rows = []
    for name, indices in layer_indices.items():
        counts = Counter(indices)
        total = len(indices)
        unique = len(counts)
        hit_multiple = sum(value >= 2 for value in counts.values())
        probability_values = []
        score_values = []
        if sampling_weights is not None and name in sampling_weights:
            probability = torch.as_tensor(
                sampling_weights[name], dtype=torch.float64
            )
            probability_values = [float(probability[index]) for index in indices]
        if scores is not None and name in scores:
            score = torch.as_tensor(scores[name], dtype=torch.float64)
            score_values = [float(score[index]) for index in indices]
        rows.append(
            {
                "event": "rcc_independent_counterfactual" if counterfactual
                else "rcc_actual",
                "layer_name": name,
                "num_filters": layer_sizes[name],
                "total_reset_slots": total,
                "unique_reset_filters": unique,
                "duplicate_slots": total - unique,
                "fraction_unique": unique / total if total else 1.0,
                "fraction_affected_filters_hit_by_multiple_tasks": (
                    hit_multiple / unique if unique else 0.0
                ),
                "mean_multiplicity": total / unique if unique else 0.0,
                "max_multiplicity": max(counts.values()) if counts else 0,
                "mean_selected_probability": (
                    sum(probability_values) / len(probability_values)
                    if probability_values else None
                ),
                "min_selected_probability": (
                    min(probability_values) if probability_values else None
                ),
                "max_selected_probability": (
                    max(probability_values) if probability_values else None
                ),
                "mean_selected_q": (
                    sum(score_values) / len(score_values) if score_values else None
                ),
            }
        )
    return rows


def independent_counterfactual_traces(
    net_glob, round_idx, task_count, args, sampling_weights
):
    """Replay independent index draws with private RNGs and no model mutation."""
    conv_layers = [
        (name, module) for name, module in net_glob.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    traces = []
    total_layers = len(conv_layers)
    for task_id in range(int(task_count)):
        seed = (
            int(args.seed) + 200000 + int(round_idx) * 100003
            + task_id * 997
        )
        rng = random.Random(seed)
        layers = []
        for depth, (name, layer) in enumerate(conv_layers):
            if round_idx >= (depth + 1) * float(args.FP_conv) / total_layers:
                continue
            num_kernels = int(layer.weight.shape[0])
            num_reset = min(
                num_kernels, int(num_kernels * float(args.reset))
            )
            weights = (
                sampling_weights.get(name)
                if sampling_weights is not None else None
            )
            indices = _sample_kernel_indices(
                num_kernels, num_reset, weights, rng
            )
            layers.append(
                {
                    "name": name,
                    "num_kernels": num_kernels,
                    "reset_indices": indices,
                }
            )
        traces.append({"seed": seed, "layers": layers})
    return traces
