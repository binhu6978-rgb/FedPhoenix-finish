from collections import defaultdict
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from Algorithm.repeatability_guidance import RepeatabilityGuidance


def _guidance(filters=2):
    return RepeatabilityGuidance(
        nn.Sequential(nn.Conv2d(1, filters, 1, bias=False))
    )


def _set_contributions(guidance, contributions, counts=None):
    """Populate exact sufficient statistics from per-client (R_i, E_i)."""
    layer_name, _key, filters = guidance.layers[0]
    guidance._stats = defaultdict(dict)
    if counts is None:
        counts = [[2] * filters for _ in contributions]
    for client_id, ((repeat, energy), client_counts) in enumerate(
        zip(contributions, counts)
    ):
        count = torch.tensor(client_counts, dtype=torch.long)
        pair_count = count.double() * (count.double() - 1.0) / 2.0
        guidance._stats[client_id][layer_name] = {
            "sum_r": torch.zeros((filters, 1, 1, 1), dtype=torch.float32),
            "count": count,
            "pair_dot": torch.tensor(repeat, dtype=torch.float64) * pair_count,
            "energy_sum": (
                torch.tensor(energy, dtype=torch.float64) * count.double()
            ),
        }


def test_identical_client_contributions_have_zero_se_and_s_equals_q():
    guidance = _guidance(filters=1)
    _set_contributions(
        guidance,
        [([0.4], [0.8]), ([0.4], [0.8]), ([0.4], [0.8])],
    )
    raw, adjusted, diagnostics = (
        guidance.finalize_window_with_jackknife_stability()
    )
    assert raw["0"].item() == pytest.approx(0.5)
    assert diagnostics["0"]["jackknife_se"].item() == pytest.approx(0.0)
    assert adjusted["0"].item() == pytest.approx(raw["0"].item())


def test_influential_client_produces_positive_se_and_shrinks_score():
    guidance = _guidance(filters=1)
    _set_contributions(
        guidance,
        [([0.9], [1.0]), ([0.1], [1.0]), ([0.1], [1.0])],
    )
    raw, adjusted, diagnostics = (
        guidance.finalize_window_with_jackknife_stability()
    )
    assert diagnostics["0"]["jackknife_se"].item() > 0.0
    assert 0.0 <= adjusted["0"].item() < raw["0"].item()


def test_helper_raw_q_exactly_matches_original_finalize_window():
    original = _guidance(filters=3)
    helper = _guidance(filters=3)
    contributions = [
        ([0.4, -0.2, 0.9], [0.8, 0.5, 1.0]),
        ([0.2, 0.3, 0.1], [0.5, 0.6, 0.7]),
        ([0.7, 0.4, 0.2], [0.9, 0.8, 0.4]),
    ]
    counts = [[2, 3, 1], [4, 2, 3], [3, 5, 2]]
    _set_contributions(original, contributions, counts)
    _set_contributions(helper, contributions, counts)
    expected = original.finalize_window()
    raw, adjusted, diagnostics = (
        helper.finalize_window_with_jackknife_stability()
    )
    assert torch.equal(raw["0"], expected["0"])
    assert adjusted["0"][0] >= 0
    assert adjusted["0"][1] >= 0
    assert adjusted["0"][2] >= 0  # two eligible clients are sufficient
    assert diagnostics["0"]["eligible_clients"].tolist() == [3, 3, 2]


def test_jackknife_helper_does_not_change_any_rng_state():
    guidance = _guidance(filters=2)
    _set_contributions(
        guidance,
        [([0.4, 0.2], [0.8, 0.7]), ([0.3, 0.5], [0.9, 0.8])],
    )
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    guidance.finalize_window_with_jackknife_stability()
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_fewer_than_two_eligible_clients_forces_adjusted_score_to_zero():
    guidance = _guidance(filters=1)
    _set_contributions(guidance, [([0.8], [1.0])])
    raw, adjusted, diagnostics = (
        guidance.finalize_window_with_jackknife_stability()
    )
    assert raw["0"].item() == pytest.approx(0.8)
    assert diagnostics["0"]["eligible_clients"].item() == 1
    assert adjusted["0"].item() == 0.0
