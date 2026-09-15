import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from Algorithm.Phoenix_util import reset_kernels_for_task
from Algorithm.timing_coverage_variants import (
    OneWindowExtensionController,
    build_coordinated_fedphoenix_tasks,
    build_rge_fedphoenix_tasks,
    independent_counterfactual_traces,
)


def _model(filters=64):
    return nn.Sequential(
        nn.Conv2d(3, filters, 3, padding=1, bias=False),
        nn.Conv2d(filters, filters, 3, padding=1, bias=False),
    )


def _args(reset=1 / 64, fp_conv=100):
    return SimpleNamespace(
        seed=1,
        reset=reset,
        remethod="ori_normal",
        FP_conv=fp_conv,
    )


def test_optional_reset_controls_leave_default_path_exact():
    model = _model()
    plain = copy.deepcopy(model)
    explicit = copy.deepcopy(model)
    trace_plain = reset_kernels_for_task(
        plain, 1 / 64, 123, at_least_one=False,
        current_iter=0, conv_transition_period=100,
    )
    trace_explicit = reset_kernels_for_task(
        explicit, 1 / 64, 123, at_least_one=False,
        current_iter=0, conv_transition_period=100,
        forced_active_layers=set(), reset_indices_by_layer={},
    )
    assert trace_plain == trace_explicit
    for first, second in zip(plain.parameters(), explicit.parameters()):
        assert torch.equal(first, second)


def test_rge_controller_allows_exactly_one_window_once():
    layers = [("early", "early.weight", 64), ("late", "late.weight", 64)]
    controller = OneWindowExtensionController(
        layers, conv_transition_period=40, rg_interval=20, threshold=0.5
    )
    controller.record_refresh(
        10,
        {"early": torch.full((64,), 0.2), "late": torch.full((64,), 0.1)},
        {"early": torch.full((64,), 1 / 64), "late": torch.full((64,), 1 / 64)},
    )
    controller.record_refresh(
        20,
        {"early": torch.full((64,), 0.12), "late": torch.full((64,), 0.1)},
        {"early": torch.full((64,), 1 / 64), "late": torch.full((64,), 1 / 64)},
    )
    active, decisions = controller.forced_active_layers(20)
    assert active == {"early"}
    assert len(decisions) == 1
    assert decisions[0]["retention_ratio"] == pytest.approx(0.6)
    assert decisions[0]["extended_rounds"] == 20
    for current_iter in range(21, 40):
        active, decisions = controller.forced_active_layers(current_iter)
        assert active == {"early"}
        assert decisions == []
    active, decisions = controller.forced_active_layers(40)
    assert "early" not in active
    assert [decision["layer_name"] for decision in decisions] == ["late"]


def test_rge_builder_changes_only_the_forced_layer_schedule():
    model = _model()
    args = _args(fp_conv=100)
    models, traces = build_rge_fedphoenix_tasks(
        model, round_idx=50, task_count=2, args=args,
        sampling_weights=None, forced_active_layers={"0"},
    )
    assert len(models) == 2
    assert all([row["name"] for row in trace["layers"]] == ["0", "1"] for trace in traces)
    assert all(len(row["reset_indices"]) == 1 for trace in traces for row in trace["layers"])


def test_rcc_preserves_task_budgets_and_adds_guided_unique_coverage():
    model = _model()
    args = _args()
    weights = {
        "0": torch.linspace(1.0, 2.0, 64),
        "1": torch.linspace(2.0, 1.0, 64),
    }
    _models, traces = build_coordinated_fedphoenix_tasks(
        model, round_idx=0, task_count=10, args=args,
        sampling_weights=weights,
    )
    for layer_name in ("0", "1"):
        selections = [
            next(row for row in trace["layers"] if row["name"] == layer_name)[
                "reset_indices"
            ]
            for trace in traces
        ]
        assert all(len(indices) == 1 for indices in selections)
        assert len({indices[0] for indices in selections}) == 10


def test_rcc_is_exact_rg_before_guidance_and_does_not_consume_global_rngs():
    model = _model()
    args = _args()
    random.seed(77)
    np.random.seed(77)
    torch.manual_seed(77)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    models, traces = build_coordinated_fedphoenix_tasks(
        model, round_idx=0, task_count=3, args=args, sampling_weights=None
    )
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    for task_id, (built_model, trace) in enumerate(zip(models, traces)):
        expected_model = copy.deepcopy(model)
        seed = args.seed + 200000 + task_id * 997
        expected_trace = reset_kernels_for_task(
            expected_model, args.reset, seed, layer_scope="all",
            init_method=args.remethod, at_least_one=False,
            current_iter=0, conv_transition_period=args.FP_conv,
            sampling_weights=None,
        )
        assert trace == expected_trace
        for first, second in zip(built_model.parameters(), expected_model.parameters()):
            assert torch.equal(first, second)


def test_independent_counterfactual_replays_original_task_indices():
    model = _model()
    args = _args()
    weights = {"0": torch.linspace(1.0, 2.0, 64), "1": torch.ones(64)}
    traces = independent_counterfactual_traces(
        model, 0, 3, args, weights
    )
    for task_id, trace in enumerate(traces):
        expected = reset_kernels_for_task(
            copy.deepcopy(model), args.reset,
            args.seed + 200000 + task_id * 997,
            init_method=args.remethod, at_least_one=False,
            current_iter=0, conv_transition_period=args.FP_conv,
            sampling_weights=weights,
        )
        assert [
            (row["name"], row["num_kernels"], row["reset_indices"])
            for row in trace["layers"]
        ] == [
            (row["name"], row["num_kernels"], row["reset_indices"])
            for row in expected["layers"]
        ]
