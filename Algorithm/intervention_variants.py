"""Orthogonal intervention-cycle variants for FedPhoenixRG experiments."""

from __future__ import annotations

import copy
import random

import torch
import torch.nn as nn

from Algorithm.Phoenix_util import _sample_kernel_indices
from models.Fed import Aggregation


def build_permuted_fedphoenix_tasks(
    net_glob,
    round_idx,
    task_count,
    reset_ratio,
    base_seed,
    conv_transition_period,
    sampling_weights=None,
):
    """Build RG-identical reset selections with within-filter permutations."""
    conv_layers = [
        (name, module) for name, module in net_glob.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    if not conv_layers:
        raise ValueError("the model has no convolutional layers")
    task_models = []
    task_traces = []
    total_conv = len(conv_layers)
    for task_id in range(int(task_count)):
        task_model = copy.deepcopy(net_glob).to("cpu")
        task_seed = (
            int(base_seed) + 200000 + int(round_idx) * 100003
            + int(task_id) * 997
        )
        python_rng = random.Random(task_seed)
        torch_rng = torch.Generator(device="cpu")
        torch_rng.manual_seed(task_seed)
        trace = {
            "seed": task_seed,
            "layer_scope": "all",
            "requested_reset_ratio": float(reset_ratio),
            "init_method": "within_filter_permutation",
            "current_iter": int(round_idx),
            "conv_transition_period": float(conv_transition_period),
            "layers": [],
        }
        task_layer_map = dict(task_model.named_modules())
        with torch.no_grad():
            for depth, (name, _global_layer) in enumerate(conv_layers):
                stop_iter = (depth + 1) * (
                    float(conv_transition_period) / total_conv
                )
                if int(round_idx) >= stop_iter:
                    continue
                layer = task_layer_map[name]
                num_kernels = int(layer.weight.shape[0])
                num_reset = min(
                    num_kernels, int(num_kernels * float(reset_ratio))
                )
                layer_weights = (
                    sampling_weights.get(name)
                    if sampling_weights is not None else None
                )
                reset_indices = _sample_kernel_indices(
                    num_kernels, num_reset, layer_weights, python_rng
                )
                relative_norms = []
                cosines = []
                fixed_fractions = []
                for index in reset_indices:
                    original = layer.weight[index].detach().clone()
                    flat = original.flatten()
                    permutation = torch.randperm(
                        flat.numel(), generator=torch_rng, device="cpu"
                    )
                    permuted = flat[permutation].reshape_as(original)
                    layer.weight[index].copy_(permuted)
                    original_double = original.double().flatten()
                    permuted_double = permuted.double().flatten()
                    original_norm = float(original_double.norm())
                    relative_norms.append(
                        float((permuted_double - original_double).norm())
                        / max(original_norm, 1e-12)
                    )
                    denominator = original_norm * float(permuted_double.norm())
                    cosines.append(
                        float((original_double * permuted_double).sum())
                        / max(denominator, 1e-12)
                    )
                    fixed_fractions.append(
                        float((permutation == torch.arange(flat.numel())).double().mean())
                    )
                trace["layers"].append(
                    {
                        "name": name,
                        "num_kernels": num_kernels,
                        "reset_indices": reset_indices,
                        "actual_reset_ratio": (
                            num_reset / num_kernels if num_kernels else 0.0
                        ),
                        "relative_perturbation_norms": relative_norms,
                        "original_permuted_cosines": cosines,
                        "fixed_position_fractions": fixed_fractions,
                    }
                )
        trace["num_reset_layers"] = len(trace["layers"])
        trace["num_reset_kernels"] = sum(
            len(layer["reset_indices"]) for layer in trace["layers"]
        )
        task_models.append(task_model)
        task_traces.append(trace)
    return task_models, task_traces


def summarize_permutation_traces(task_traces):
    """Aggregate direct permutation diagnostics by layer for one round."""
    accumulators = {}
    for trace in task_traces:
        for layer in trace.get("layers", []):
            target = accumulators.setdefault(
                layer["name"], {"relative": [], "cosine": [], "fixed": []}
            )
            target["relative"].extend(layer["relative_perturbation_norms"])
            target["cosine"].extend(layer["original_permuted_cosines"])
            target["fixed"].extend(layer["fixed_position_fractions"])
    return {
        name: {
            "reset_observations": len(values["relative"]),
            "mean_relative_perturbation_norm": (
                sum(values["relative"]) / len(values["relative"])
                if values["relative"] else 0.0
            ),
            "mean_original_permuted_cosine": (
                sum(values["cosine"]) / len(values["cosine"])
                if values["cosine"] else 0.0
            ),
            "mean_fixed_position_fraction": (
                sum(values["fixed"]) / len(values["fixed"])
                if values["fixed"] else 0.0
            ),
        }
        for name, values in accumulators.items()
    }


@torch.no_grad()
def aggregate_from_actual_task_deltas(
    local_states,
    lens,
    global_before_reset,
    task_models,
    task_traces,
):
    """Sample-weight aggregate local deltas from each task's actual start."""
    if not (
        len(local_states) == len(lens) == len(task_models) == len(task_traces)
    ):
        raise ValueError("local states, weights, task models, and traces must align")
    aggregated = Aggregation(local_states, lens)
    total_weight = float(sum(lens))
    if total_weight <= 0.0:
        raise ValueError("aggregation weight must be positive")
    task_starts = [model.state_dict() for model in task_models]
    diagnostics = {}
    correction_sums = {}
    reset_counts = {}
    for task_id, (local_state, task_start, trace, sample_count) in enumerate(
        zip(local_states, task_starts, task_traces, lens)
    ):
        client_weight = float(sample_count) / total_weight
        for layer in trace.get("layers", []):
            indices = layer.get("reset_indices", [])
            if not indices:
                continue
            name = layer["name"]
            state_key = f"{name}.weight"
            index = torch.as_tensor(indices, dtype=torch.long)
            device = aggregated[state_key].device
            index_device = index.to(device=device)
            global_filter = global_before_reset[state_key][index].detach().to(
                device=device, dtype=aggregated[state_key].dtype
            )
            reset_filter = task_start[state_key][index].detach().to(
                device=device, dtype=aggregated[state_key].dtype
            )
            local_filter = local_state[state_key][index_device].detach().to(
                device=device, dtype=aggregated[state_key].dtype
            )
            correction = global_filter - reset_filter
            aggregated[state_key][index_device] += client_weight * correction

            perturbation = (reset_filter - global_filter).double().flatten(1)
            local_step = (local_filter - reset_filter).double().flatten(1)
            perturbation_norm = perturbation.norm(dim=1)
            local_step_norm = local_step.norm(dim=1)
            cosine = (perturbation * local_step).sum(dim=1) / (
                perturbation_norm * local_step_norm
            ).clamp_min(1e-12)
            target = diagnostics.setdefault(
                name,
                {
                    "reset_observations": 0,
                    "perturbation_norm_sum": 0.0,
                    "local_step_norm_sum": 0.0,
                    "step_over_perturbation_sum": 0.0,
                    "cosine_sum": 0.0,
                },
            )
            count = len(indices)
            target["reset_observations"] += count
            target["perturbation_norm_sum"] += float(perturbation_norm.sum())
            target["local_step_norm_sum"] += float(local_step_norm.sum())
            target["step_over_perturbation_sum"] += float(
                (local_step_norm / perturbation_norm.clamp_min(1e-12)).sum()
            )
            target["cosine_sum"] += float(cosine.sum())

            if name not in correction_sums:
                correction_sums[name] = torch.zeros_like(
                    aggregated[state_key], device=device
                )
                reset_counts[name] = torch.zeros(
                    aggregated[state_key].shape[0], dtype=torch.long, device=device
                )
            correction_sums[name][index_device] += client_weight * correction
            reset_counts[name][index_device] += 1

    for name, target in diagnostics.items():
        count = target.pop("reset_observations")
        correction = correction_sums[name]
        incidence = reset_counts[name]
        target.update(
            {
                "reset_observations": count,
                "mean_perturbation_norm": target.pop("perturbation_norm_sum") / count,
                "mean_local_step_norm": target.pop("local_step_norm_sum") / count,
                "mean_step_over_perturbation": (
                    target.pop("step_over_perturbation_sum") / count
                ),
                "mean_perturbation_step_cosine": target.pop("cosine_sum") / count,
                "aggregate_correction_norm": float(correction.double().norm()),
                "unique_reset_filters": int((incidence > 0).sum()),
                "multi_client_filter_fraction": float(
                    (incidence[incidence > 0] >= 2).double().mean()
                ),
            }
        )
    return aggregated, diagnostics
