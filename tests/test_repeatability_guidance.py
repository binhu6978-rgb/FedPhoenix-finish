import random

import numpy as np
import torch
import torch.nn as nn

from Algorithm.repeatability_guidance import (
    RepeatabilityGuidance,
    layer_calibrated_sampling_weights,
    remap_repeatability_scores,
    sampling_weights_from_scores,
)


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


def test_mixture_probabilities_are_normalized_and_nonnegative():
    score = torch.tensor([0.0, 1.0, 3.0, 0.0])
    probability = sampling_weights_from_scores({"layer": score}, 0.5)["layer"]
    expected = 0.5 * torch.full((4,), 0.25, dtype=torch.float64)
    expected += 0.5 * score.double() / score.sum()
    assert torch.allclose(probability, expected)
    assert torch.isclose(probability.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert bool((probability >= 0).all())


def test_zero_scores_fall_back_to_uniform_probabilities():
    probability = sampling_weights_from_scores(
        {"layer": torch.zeros(5)}, 0.75
    )["layer"]
    assert torch.equal(probability, torch.full((5,), 0.2, dtype=torch.float64))


def test_layer_calibration_uses_reset_budget_weighted_signal_strength():
    scores = {
        "weak": torch.tensor([0.0, 0.2]),
        "strong": torch.tensor([0.2, 0.6]),
    }
    weights, diagnostics = layer_calibrated_sampling_weights(
        scores, {"weak": 1, "strong": 3}, base_mix=0.75
    )

    assert abs(diagnostics["s_bar"] - 0.325) < 1e-7
    assert abs(diagnostics["gamma"]["weak"] - 0.75 * 0.1 / 0.325) < 1e-7
    assert abs(diagnostics["gamma"]["strong"] - 0.75) < 1e-7
    for probability in weights.values():
        assert torch.isclose(
            probability.sum(), torch.tensor(1.0, dtype=torch.float64)
        )
        assert bool((probability >= 0).all())


def test_layer_calibration_uniform_fallback_and_eligible_diagnostics():
    model = _model(filters=3)
    guidance = RepeatabilityGuidance(model)
    global_state = _state(model, [0.0, 0.0, 0.0])
    local_states = [
        _state(model, [1.0, 2.0, 3.0]),
        _state(model, [-1.0, -2.0, -3.0]),
    ]
    guidance.observe_round(global_state, local_states, [0, 1], [_trace(0), _trace()])
    guidance.observe_round(global_state, local_states, [0, 1], [_trace(0), _trace()])
    scores = guidance.finalize_window()
    eligible = guidance.get_layer_diagnostics()["0"]
    probability, calibration = layer_calibrated_sampling_weights(
        {"0": torch.zeros(3)}, {"0": 1}, base_mix=0.75
    )

    assert scores["0"].shape == (3,)
    assert eligible["mean_eligible_clients"] == 4 / 3
    assert eligible["min_eligible_clients"] == 0
    assert calibration["gamma"]["0"] == 0.0
    assert torch.equal(
        probability["0"], torch.full((3,), 1 / 3, dtype=torch.float64)
    )


def test_excess_remap_uses_only_above_layer_mean_scores():
    scores = {"layer": torch.tensor([0.0, 0.2, 0.6, 0.8])}
    excess = remap_repeatability_scores(scores, "excess")
    expected_score = torch.tensor([0.0, 0.0, 0.2, 0.4], dtype=torch.float64)
    assert torch.allclose(excess["layer"], expected_score, atol=1e-7)
    probability = sampling_weights_from_scores(excess, 0.75)["layer"]
    expected_probability = 0.25 * torch.full((4,), 0.25, dtype=torch.float64)
    expected_probability += 0.75 * expected_score / expected_score.sum()
    assert torch.allclose(probability, expected_probability, atol=1e-7)
    flat = remap_repeatability_scores(
        {"layer": torch.full((4,), 0.3)}, "excess"
    )
    uniform = sampling_weights_from_scores(flat, 0.75)["layer"]
    assert torch.equal(uniform, torch.full((4,), 0.25, dtype=torch.float64))


def test_persistent_remap_uses_two_consecutive_windows():
    current = {"layer": torch.tensor([0.0, 0.25, 1.0, 0.5])}
    previous = {"layer": torch.tensor([1.0, 1.0, 0.25, 0.0])}
    first = remap_repeatability_scores(current, "persistent")
    assert torch.equal(
        sampling_weights_from_scores(first, 0.75)["layer"],
        sampling_weights_from_scores(current, 0.75)["layer"],
    )
    persistent = remap_repeatability_scores(current, "persistent", previous)
    assert torch.allclose(
        persistent["layer"], torch.tensor([0.0, 0.5, 0.5, 0.0], dtype=torch.float64)
    )
    probability = sampling_weights_from_scores(persistent, 0.75)["layer"]
    assert torch.allclose(
        probability, torch.tensor([0.0625, 0.4375, 0.4375, 0.0625], dtype=torch.float64)
    )
    disjoint = remap_repeatability_scores(
        {"layer": torch.tensor([1.0, 0.0])}, "persistent",
        {"layer": torch.tensor([0.0, 1.0])},
    )
    assert torch.equal(
        sampling_weights_from_scores(disjoint, 0.75)["layer"],
        torch.full((2,), 0.5, dtype=torch.float64),
    )
