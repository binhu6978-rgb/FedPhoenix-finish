"""Streaming cross-round repeatability scores for FedPhoenix reset guidance.

The accumulator stores exact sufficient statistics for all same-client pairs
within the current window.  It never samples randomness and keeps all state on
CPU, so observing or finalizing a window cannot perturb the training RNGs.
"""

from __future__ import annotations

from collections import defaultdict
import math

import torch
import torch.nn as nn


def convolution_layers(model: nn.Module):
    """Return ``(module_name, state_dict_key, output_filters)`` in model order."""
    return [
        (name, f"{name}.weight", int(module.weight.shape[0]))
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]


class RepeatabilityGuidance:
    """Accumulate filter-level client repeatability over one finite window."""

    def __init__(self, model: nn.Module, epsilon: float = 1e-12):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.layers = convolution_layers(model)
        if not self.layers:
            raise ValueError("the model has no convolutional layers")
        self.epsilon = float(epsilon)
        self._scores = {}
        self._diagnostics = {}
        self.reset_window()

    def reset_window(self):
        """Discard current-window statistics while preserving finalized scores."""
        self._stats = defaultdict(dict)
        self.rounds_observed = 0

    @staticmethod
    def _reset_masks(reset_traces, layer_name, output_filters):
        masks = torch.ones(
            (len(reset_traces), output_filters), dtype=torch.bool, device="cpu"
        )
        for task_id, trace in enumerate(reset_traces):
            for layer_trace in trace.get("layers", []):
                if layer_trace.get("name") == layer_name:
                    indices = layer_trace.get("reset_indices", [])
                    if indices:
                        masks[task_id, torch.as_tensor(indices, dtype=torch.long)] = False
                    break
        return masks

    def _client_layer_stats(self, client_id, layer_name, shape):
        client_stats = self._stats[int(client_id)]
        if layer_name not in client_stats:
            output_filters = int(shape[0])
            client_stats[layer_name] = {
                "sum_r": torch.zeros(shape, dtype=torch.float32, device="cpu"),
                "count": torch.zeros(output_filters, dtype=torch.long),
                "pair_dot": torch.zeros(output_filters, dtype=torch.float64),
                "energy_sum": torch.zeros(output_filters, dtype=torch.float64),
            }
        return client_stats[layer_name]

    @torch.no_grad()
    def observe_round(self, global_before_reset, local_states, client_ids, reset_traces):
        """Add one round, excluding every client/filter that was reset.

        ``global_before_reset`` is the true server state before task copies are
        perturbed.  Round centering is performed separately for each filter over
        only its non-reset clients.  Filters with fewer than two valid clients
        contribute no observation in this round.
        """
        if not (len(local_states) == len(client_ids) == len(reset_traces)):
            raise ValueError("local_states, client_ids, and reset_traces must align")
        if not local_states:
            self.rounds_observed += 1
            return

        for layer_name, state_key, output_filters in self.layers:
            reference = global_before_reset[state_key].detach().to(
                device="cpu", dtype=torch.float32
            )
            deltas = torch.stack(
                [
                    state[state_key].detach().to(device="cpu", dtype=torch.float32)
                    - reference
                    for state in local_states
                ]
            )
            valid = self._reset_masks(reset_traces, layer_name, output_filters)
            valid_count = valid.sum(dim=0)
            usable = valid_count >= 2
            if not bool(usable.any()):
                continue

            expand = (1,) * (deltas.ndim - 2)
            mask = valid.reshape(valid.shape + expand).to(deltas.dtype)
            denominator = valid_count.clamp_min(1).reshape((1, output_filters) + expand)
            round_mean = (deltas * mask).sum(dim=0, keepdim=True) / denominator
            relative = deltas - round_mean

            for task_id, client_id in enumerate(client_ids):
                accepted = valid[task_id] & usable
                if not bool(accepted.any()):
                    continue
                current = relative[task_id]
                stats = self._client_layer_stats(client_id, layer_name, reference.shape)
                previous = stats["sum_r"][accepted]
                now = current[accepted]
                stats["pair_dot"][accepted] += (
                    now.double().flatten(1) * previous.double().flatten(1)
                ).sum(dim=1)
                stats["sum_r"][accepted] = previous + now
                stats["energy_sum"][accepted] += (
                    now.double().flatten(1).square().sum(dim=1)
                )
                stats["count"][accepted] += 1

        self.rounds_observed += 1

    @torch.no_grad()
    def finalize_window(self):
        """Compute client-balanced ``q_g = clip([R_g]+/(E_g+eps), 0, 1)``."""
        scores = {}
        diagnostics = {}
        for layer_name, _state_key, output_filters in self.layers:
            repeat_sum = torch.zeros(output_filters, dtype=torch.float64)
            energy_sum = torch.zeros(output_filters, dtype=torch.float64)
            eligible_clients = torch.zeros(output_filters, dtype=torch.long)
            for client_stats in self._stats.values():
                stats = client_stats.get(layer_name)
                if stats is None:
                    continue
                count = stats["count"]
                eligible = count >= 2
                if not bool(eligible.any()):
                    continue
                n = count[eligible].double()
                pair_count = n * (n - 1.0) / 2.0
                repeat_sum[eligible] += stats["pair_dot"][eligible] / pair_count
                energy_sum[eligible] += stats["energy_sum"][eligible] / n
                eligible_clients[eligible] += 1

            present = eligible_clients > 0
            repeatability = torch.zeros(output_filters, dtype=torch.float64)
            energy = torch.zeros(output_filters, dtype=torch.float64)
            repeatability[present] = (
                repeat_sum[present] / eligible_clients[present].double()
            )
            energy[present] = energy_sum[present] / eligible_clients[present].double()
            score = torch.zeros(output_filters, dtype=torch.float32)
            score[present] = (
                repeatability[present].clamp_min(0.0)
                / (energy[present] + self.epsilon)
            ).clamp_(0.0, 1.0).float()
            scores[layer_name] = score
            diagnostics[layer_name] = {
                "mean_eligible_clients": float(eligible_clients.double().mean()),
                "min_eligible_clients": int(eligible_clients.min()),
            }
        self._scores = scores
        self._diagnostics = diagnostics
        return self.get_layer_scores()

    def get_layer_scores(self):
        """Return independent copies of the most recently finalized q scores."""
        return {name: score.clone() for name, score in self._scores.items()}

    def get_layer_diagnostics(self):
        """Return eligible-client summaries from the last finalized window."""
        return {name: dict(values) for name, values in self._diagnostics.items()}


def sampling_weights_from_scores(scores, mix, epsilon=1e-12):
    """Mix uniform reset placement with normalized repeatability scores.

    The returned tensors sum to one and are used as relative weights by the
    task-private weighted sampler.  A layer with no positive score follows the
    exact uniform FedPhoenix path.
    """
    mix = float(mix)
    if not 0.0 <= mix <= 1.0:
        raise ValueError("rg_mix must be in [0, 1]")
    weights = {}
    for name, raw_score in scores.items():
        score = torch.as_tensor(raw_score, dtype=torch.float64).clamp_min(0.0)
        if score.numel() == 0:
            raise ValueError("repeatability score tensors must be non-empty")
        uniform = torch.full_like(score, 1.0 / score.numel())
        total = score.sum()
        if not torch.isfinite(total) or float(total) <= float(epsilon):
            probability = uniform
        else:
            probability = (1.0 - mix) * uniform + mix * (score / total)
        weights[name] = probability
    return weights


def remap_repeatability_scores(scores, mode, previous_scores=None):
    """Remap finalized q without changing its estimator or window statistics."""
    if mode not in {"excess", "persistent"}:
        raise ValueError("mode must be excess or persistent")
    remapped = {}
    for name, raw_score in scores.items():
        score = torch.as_tensor(raw_score, dtype=torch.float64).clamp_min(0.0)
        if mode == "excess":
            remapped[name] = (score - score.mean()).clamp_min(0.0)
        elif previous_scores is None:
            remapped[name] = score.clone()
        else:
            if name not in previous_scores:
                raise ValueError(f"previous scores missing layer {name}")
            previous = torch.as_tensor(
                previous_scores[name], dtype=torch.float64
            ).clamp_min(0.0)
            if previous.shape != score.shape:
                raise ValueError(f"previous score shape differs for {name}")
            remapped[name] = (score * previous).sqrt()
    return remapped


def layer_calibrated_sampling_weights(
    scores, reset_counts, base_mix=0.75, epsilon=1e-12
):
    """Apply reset-budget-weighted layer calibration to the existing q scores."""
    base_mix = float(base_mix)
    if not 0.0 <= base_mix <= 1.0:
        raise ValueError("base_mix must be in [0, 1]")
    mean_q = {
        name: float(torch.as_tensor(score, dtype=torch.float64).mean())
        for name, score in scores.items()
    }
    active_budget = {
        name: int(count)
        for name, count in reset_counts.items()
        if int(count) > 0 and name in scores
    }
    total_budget = sum(active_budget.values())
    s_bar = (
        sum(active_budget[name] * mean_q[name] for name in active_budget)
        / total_budget
        if total_budget
        else 0.0
    )

    weights = {}
    gamma = {}
    for name, raw_score in scores.items():
        score = torch.as_tensor(raw_score, dtype=torch.float64).clamp_min(0.0)
        if score.numel() == 0:
            raise ValueError("repeatability score tensors must be non-empty")
        uniform = torch.full_like(score, 1.0 / score.numel())
        total = score.sum()
        layer_gamma = 0.0
        if (
            name in active_budget
            and math.isfinite(s_bar)
            and s_bar > float(epsilon)
            and torch.isfinite(total)
            and float(total) > float(epsilon)
        ):
            layer_gamma = base_mix * min(
                1.0, mean_q[name] / (s_bar + float(epsilon))
            )
            probability = (
                (1.0 - layer_gamma) * uniform
                + layer_gamma * (score / total)
            )
        else:
            probability = uniform
        weights[name] = probability
        gamma[name] = float(layer_gamma)
    return weights, {"s_bar": float(s_bar), "mean_q": mean_q, "gamma": gamma}
