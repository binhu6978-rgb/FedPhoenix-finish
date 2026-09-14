import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import main_fed
from Algorithm.Phoenix_util import reset_kernels_for_task
from Algorithm.repeatability_guidance import RepeatabilityGuidance


_ORIGINAL_BUILD_TASKS = main_fed._build_fedphoenix_tasks


def _two_layer_model():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, bias=False),
        nn.Conv2d(8, 16, 3, bias=False),
    )


def test_original_fedphoenix_reset_is_unchanged_when_weights_are_none():
    base = _two_layer_model()
    omitted = copy.deepcopy(base)
    explicit = copy.deepcopy(base)
    trace_omitted = reset_kernels_for_task(
        omitted, 0.375, 123, at_least_one=False
    )
    trace_explicit = reset_kernels_for_task(
        explicit, 0.375, 123, at_least_one=False, sampling_weights=None
    )
    assert trace_omitted == trace_explicit
    assert [layer["reset_indices"] for layer in trace_explicit["layers"]] == [
        [0, 2, 7],
        [0, 1, 4, 6, 8, 13],
    ]
    for key, value in omitted.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])


def test_guidance_keeps_reset_budget_and_stair_schedule():
    base = _two_layer_model()
    weights = {"0": torch.arange(1, 9), "1": torch.arange(1, 17)}
    for round_id in [0, 30, 60, 100]:
        plain = reset_kernels_for_task(
            copy.deepcopy(base), 0.25, 44,
            at_least_one=False, current_iter=round_id, conv_transition_period=100,
        )
        guided = reset_kernels_for_task(
            copy.deepcopy(base), 0.25, 44,
            at_least_one=False, current_iter=round_id, conv_transition_period=100,
            sampling_weights=weights,
        )
        assert guided["num_reset_layers"] == plain["num_reset_layers"]
        assert guided["num_reset_kernels"] == plain["num_reset_kernels"]
        assert [x["name"] for x in guided["layers"]] == [
            x["name"] for x in plain["layers"]
        ]
        assert [len(x["reset_indices"]) for x in guided["layers"]] == [
            len(x["reset_indices"]) for x in plain["layers"]
        ]


def test_lc_calibration_budget_matches_original_reset_schedule():
    base = _two_layer_model()
    layers = RepeatabilityGuidance(base).layers
    args = SimpleNamespace(reset=0.25, FP_conv=100)
    for round_id in [0, 30, 60, 100]:
        expected = reset_kernels_for_task(
            copy.deepcopy(base), 0.25, 44,
            at_least_one=False, current_iter=round_id,
            conv_transition_period=100,
        )
        actual = main_fed._active_fedphoenix_reset_counts(
            layers, round_id, args
        )
        assert actual == {
            layer["name"]: len(layer["reset_indices"])
            for layer in expected["layers"]
        }


def test_guided_tasks_keep_independent_reset_patterns():
    model = nn.Sequential(nn.Conv2d(3, 64, 3, bias=False))
    args = SimpleNamespace(seed=5, reset=0.125, remethod="ori_normal", FP_conv=100)
    weights = {"0": torch.linspace(1.0, 2.0, 64)}
    _models, traces = main_fed._build_fedphoenix_tasks(
        model, 0, 5, args, sampling_weights=weights
    )
    patterns = [tuple(trace["layers"][0]["reset_indices"]) for trace in traces]
    assert len(set(patterns)) == len(patterns)
    assert all(len(pattern) == 8 for pattern in patterns)


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=1, bias=False)


class _FakeLocalUpdate:
    def __init__(self, args, dataset, idxs, dataset_test):
        self.shift = (int(next(iter(idxs))) + 1) * 1e-4

    def train(self, net):
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.add_(self.shift)
        return net.state_dict()


def _run(method, initial_state, monkeypatch, epochs=20):
    args = SimpleNamespace(
        epochs=epochs,
        frac=0.5,
        num_users=6,
        seed=31,
        reset=0.25,
        remethod="ori_normal",
        FP_conv=100,
        device=torch.device("cpu"),
        algorithm=method,
        rg_interval=20,
        rg_mix=0.5,
        dataset="synthetic",
        model="tiny",
        run_name="",
        metrics_log_dir="unused",
    )
    captured = []
    reset_traces = []
    def record_tasks(*task_args, **task_kwargs):
        models, traces = _ORIGINAL_BUILD_TASKS(*task_args, **task_kwargs)
        reset_traces.append([
            [tuple(layer["reset_indices"]) for layer in trace["layers"]]
            for trace in traces
        ])
        return models, traces

    monkeypatch.setattr(main_fed, "args", args, raising=False)
    monkeypatch.setattr(main_fed, "_build_fedphoenix_tasks", record_tasks)
    monkeypatch.setattr(main_fed, "LocalUpdate_FedAvg", _FakeLocalUpdate)
    monkeypatch.setattr(
        main_fed,
        "evaluate_round_accuracy",
        lambda model, dataset, args, round_id: float(
            sum(x.double().sum() for x in model.state_dict().values())
        ),
    )
    monkeypatch.setattr(main_fed, "print_peak_accuracy", lambda *items: None)
    monkeypatch.setattr(
        main_fed, "_write_training_metrics", lambda args, rows: captured.extend(rows)
    )
    model = _TinyNet()
    model.load_state_dict(copy.deepcopy(initial_state))
    users = {client: [client] for client in range(args.num_users)}
    np.random.seed(args.seed)
    if method == "FedPhoenix":
        main_fed.FedPhoenix(model, None, None, None, users)
    elif method == "FedPhoenixRG-AA":
        main_fed.FedPhoenixRGAA(model, None, None, None, users)
    elif method == "FedPhoenixRG-Recovery":
        main_fed.FedPhoenixRGRecovery(model, None, None, None, users)
    elif method == "FedPhoenixRG-Excess":
        main_fed.FedPhoenixRGRemapped(
            model, None, None, None, users, "excess"
        )
    elif method == "FedPhoenixRG-Persistent":
        main_fed.FedPhoenixRGRemapped(
            model, None, None, None, users, "persistent"
        )
    elif method == "FedPhoenixRG-LC":
        main_fed.FedPhoenixRGLC(model, None, None, None, users)
    else:
        main_fed.FedPhoenixRG(model, None, None, None, users)
    return copy.deepcopy(model.state_dict()), captured, reset_traces


def test_first_twenty_rounds_are_exactly_fedphoenix(monkeypatch):
    torch.manual_seed(9)
    initial_state = _TinyNet().state_dict()
    baseline_state, baseline_rows, baseline_traces = _run(
        "FedPhoenix", initial_state, monkeypatch
    )
    guided_state, guided_rows, guided_traces = _run(
        "FedPhoenixRG", initial_state, monkeypatch
    )

    assert [row["selected_clients"] for row in baseline_rows] == [
        row["selected_clients"] for row in guided_rows
    ]
    assert [row["task_seeds"] for row in baseline_rows] == [
        row["task_seeds"] for row in guided_rows
    ]
    assert baseline_traces == guided_traces
    assert [row["test_accuracy"] for row in baseline_rows] == [
        row["test_accuracy"] for row in guided_rows
    ]
    for key, value in baseline_state.items():
        assert torch.equal(value, guided_state[key]), key


def test_lc_first_twenty_rounds_are_exactly_fedphoenix(monkeypatch):
    torch.manual_seed(9)
    initial_state = _TinyNet().state_dict()
    baseline_state, baseline_rows, baseline_traces = _run(
        "FedPhoenix", initial_state, monkeypatch
    )
    lc_state, lc_rows, lc_traces = _run(
        "FedPhoenixRG-LC", initial_state, monkeypatch
    )

    assert [row["selected_clients"] for row in baseline_rows] == [
        row["selected_clients"] for row in lc_rows
    ]
    assert [row["task_seeds"] for row in baseline_rows] == [
        row["task_seeds"] for row in lc_rows
    ]
    assert baseline_traces == lc_traces
    assert [row["test_accuracy"] for row in baseline_rows] == [
        row["test_accuracy"] for row in lc_rows
    ]
    for key, value in baseline_state.items():
        assert torch.equal(value, lc_state[key]), key


def test_remapped_methods_preserve_rg_before_first_guidance(monkeypatch):
    torch.manual_seed(9)
    initial_state = _TinyNet().state_dict()
    _rg_state, rg_rows, rg_traces = _run(
        "FedPhoenixRG", initial_state, monkeypatch, epochs=21
    )
    for method in ("FedPhoenixRG-Excess", "FedPhoenixRG-Persistent"):
        _state, rows, traces = _run(
            method, initial_state, monkeypatch, epochs=21
        )
        assert [row["test_accuracy"] for row in rows[:20]] == [
            row["test_accuracy"] for row in rg_rows[:20]
        ]
        assert [row["selected_clients"] for row in rows] == [
            row["selected_clients"] for row in rg_rows
        ]
        assert [row["task_seeds"] for row in rows] == [
            row["task_seeds"] for row in rg_rows
        ]
        assert traces[:20] == rg_traces[:20]
        if method == "FedPhoenixRG-Persistent":
            assert traces[20] == rg_traces[20]
            assert rows[20]["test_accuracy"] == rg_rows[20]["test_accuracy"]


def test_aa_and_recovery_preserve_rg_for_first_twenty_rounds(monkeypatch):
    torch.manual_seed(9)
    initial_state = _TinyNet().state_dict()
    rg_state, rg_rows, rg_traces = _run(
        "FedPhoenixRG", initial_state, monkeypatch
    )
    for method in ("FedPhoenixRG-AA", "FedPhoenixRG-Recovery"):
        state, rows, traces = _run(method, initial_state, monkeypatch)
        assert [row["selected_clients"] for row in rows] == [
            row["selected_clients"] for row in rg_rows
        ]
        assert [row["task_seeds"] for row in rows] == [
            row["task_seeds"] for row in rg_rows
        ]
        assert traces == rg_traces
        assert [row["test_accuracy"] for row in rows] == [
            row["test_accuracy"] for row in rg_rows
        ]
        for key, value in rg_state.items():
            assert torch.equal(value, state[key]), key
