import random

import numpy as np
import torch
import torch.nn as nn

from Algorithm.repeatability_guidance import RepeatabilityGuidance


def _model(filters=2):
    return nn.Sequential(nn.Conv2d(1, filters, kernel_size=1, bias=False))


def _state(model, values):
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    state["0.weight"] = torch.as_tensor(values, dtype=torch.float32).reshape_as(
        state["0.weight"]
    )
    return state


def _trace(*reset_indices):
    return {
        "layers": [
            {"name": "0", "reset_indices": list(reset_indices)}
        ]
    }


def test_reset_observation_is_excluded_from_mean_and_accumulator():
    model = _model()
    guidance = RepeatabilityGuidance(model)
    global_state = _state(model, [0.0, 0.0])
    local_states = [
        _state(model, [100.0, 0.0]),
        _state(model, [1.0, 1.0]),
        _state(model, [3.0, 2.0]),
    ]
    traces = [_trace(0), _trace(), _trace()]

    guidance.observe_round(global_state, local_states, [0, 1, 2], traces)
    guidance.observe_round(global_state, local_states, [0, 1, 2], traces)

    assert guidance._stats[0]["0"]["count"][0].item() == 0
    assert guidance._stats[1]["0"]["count"][0].item() == 2
    assert guidance._stats[2]["0"]["count"][0].item() == 2
    assert guidance._stats[1]["0"]["sum_r"][0].item() == -2.0
    assert guidance._stats[2]["0"]["sum_r"][0].item() == 2.0
    assert guidance._stats[1]["0"]["pair_dot"][0].item() == 1.0
    assert guidance._stats[2]["0"]["pair_dot"][0].item() == 1.0


def test_persistent_filter_scores_above_nonrepeatable_filter():
    model = _model()
    guidance = RepeatabilityGuidance(model)
    global_state = _state(model, [0.0, 0.0])
    persistent = [-3.0, -1.0, 1.0, 3.0]
    changing = [
        [-3.0, -1.0, 1.0, 3.0],
        [1.0, 3.0, -3.0, -1.0],
        [3.0, -3.0, -1.0, 1.0],
        [-1.0, 1.0, 3.0, -3.0],
    ]
    for round_id in range(8):
        noise = changing[round_id % len(changing)]
        local_states = [
            _state(model, [persistent[client], noise[client]])
            for client in range(4)
        ]
        guidance.observe_round(
            global_state, local_states, list(range(4)), [_trace()] * 4
        )

    scores = guidance.finalize_window()["0"]
    assert scores[0].item() > scores[1].item()
    assert scores[0].item() > 0.9


def test_guidance_does_not_change_global_random_streams():
    model = _model()
    guidance = RepeatabilityGuidance(model)
    global_state = _state(model, [0.0, 0.0])
    local_states = [_state(model, [-1.0, 1.0]), _state(model, [1.0, -1.0])]

    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    guidance.observe_round(global_state, local_states, [0, 1], [_trace(), _trace()])
    guidance.observe_round(global_state, local_states, [0, 1], [_trace(), _trace()])
    guidance.finalize_window()
    guidance.reset_window()

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
