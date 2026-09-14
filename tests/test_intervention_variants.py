import copy
import random

import numpy as np
import torch
import torch.nn as nn

from Algorithm.Phoenix_util import reset_kernels_for_task
from Algorithm.intervention_variants import (
    aggregate_from_actual_task_deltas,
    build_permuted_fedphoenix_tasks,
)


def _model():
    return nn.Sequential(nn.Conv2d(1, 2, kernel_size=1, bias=False))


def _set_state(model, values):
    state = copy.deepcopy(model.state_dict())
    state["0.weight"] = torch.tensor(values, dtype=torch.float32).reshape_as(
        state["0.weight"]
    )
    model.load_state_dict(state)
    return model


def test_delta_aggregation_removes_reset_start_offset_at_same_client_weight():
    global_model = _set_state(_model(), [10.0, 20.0])
    reset_task = _set_state(_model(), [0.0, 20.0])
    normal_task = copy.deepcopy(global_model)
    local_states = [
        _set_state(_model(), [2.0, 21.0]).state_dict(),
        _set_state(_model(), [11.0, 22.0]).state_dict(),
    ]
    traces = [
        {"layers": [{"name": "0", "reset_indices": [0]}]},
        {"layers": [{"name": "0", "reset_indices": []}]},
    ]

    result, diagnostics = aggregate_from_actual_task_deltas(
        local_states,
        [1, 3],
        global_model.state_dict(),
        [reset_task, normal_task],
        traces,
    )

    assert torch.allclose(
        result["0.weight"].flatten(), torch.tensor([11.25, 21.75])
    )
    layer = diagnostics["0"]
    assert layer["reset_observations"] == 1
    assert layer["mean_perturbation_norm"] == 10.0
    assert layer["mean_local_step_norm"] == 2.0
    assert layer["mean_step_over_perturbation"] == 0.2
    assert layer["mean_perturbation_step_cosine"] == -1.0
    assert layer["aggregate_correction_norm"] == 2.5


def test_permutation_preserves_filter_values_and_rg_filter_selection():
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, bias=False),
        nn.Conv2d(8, 16, kernel_size=3, bias=False),
    )
    weights = {"0": torch.arange(1, 9), "1": torch.arange(1, 17)}
    plain = copy.deepcopy(model)
    plain_trace = reset_kernels_for_task(
        plain, 0.25, 123, at_least_one=False,
        current_iter=0, conv_transition_period=100,
        sampling_weights=weights,
    )
    permuted_models, permuted_traces = build_permuted_fedphoenix_tasks(
        model, round_idx=0, task_count=1, reset_ratio=0.25,
        base_seed=-199877, conv_transition_period=100,
        sampling_weights=weights,
    )
    permuted = permuted_models[0]
    permuted_trace = permuted_traces[0]

    assert permuted_trace["seed"] == 123
    assert [layer["reset_indices"] for layer in permuted_trace["layers"]] == [
        layer["reset_indices"] for layer in plain_trace["layers"]
    ]
    original_modules = dict(model.named_modules())
    permuted_modules = dict(permuted.named_modules())
    for layer_trace in permuted_trace["layers"]:
        name = layer_trace["name"]
        reset = set(layer_trace["reset_indices"])
        for index in range(layer_trace["num_kernels"]):
            original = original_modules[name].weight[index].detach().flatten()
            changed = permuted_modules[name].weight[index].detach().flatten()
            if index in reset:
                assert torch.equal(original.sort().values, changed.sort().values)
                assert torch.allclose(original.norm(), changed.norm(), atol=1e-7)
            else:
                assert torch.equal(original, changed)


def test_permutation_builder_uses_only_private_rngs():
    model = nn.Sequential(nn.Conv2d(3, 64, 3, bias=False))
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    build_permuted_fedphoenix_tasks(
        model, round_idx=0, task_count=3, reset_ratio=1 / 64,
        base_seed=1, conv_transition_period=1000,
    )

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_permutation_matches_rg_reset_budget_and_stair_schedule():
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, bias=False),
        nn.Conv2d(8, 16, kernel_size=3, bias=False),
    )
    weights = {"0": torch.arange(1, 9), "1": torch.arange(1, 17)}
    for round_idx in (0, 30, 60, 100):
        task_seed = 44
        plain = reset_kernels_for_task(
            copy.deepcopy(model), 0.25, task_seed, at_least_one=False,
            current_iter=round_idx, conv_transition_period=100,
            sampling_weights=weights,
        )
        _models, traces = build_permuted_fedphoenix_tasks(
            model, round_idx=round_idx, task_count=1, reset_ratio=0.25,
            base_seed=task_seed - 200000 - round_idx * 100003,
            conv_transition_period=100, sampling_weights=weights,
        )
        permuted = traces[0]
        assert permuted["num_reset_layers"] == plain["num_reset_layers"]
        assert permuted["num_reset_kernels"] == plain["num_reset_kernels"]
        assert [layer["name"] for layer in permuted["layers"]] == [
            layer["name"] for layer in plain["layers"]
        ]
        assert [layer["reset_indices"] for layer in permuted["layers"]] == [
            layer["reset_indices"] for layer in plain["layers"]
        ]
