from collections import defaultdict
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from Algorithm.repeatability_guidance import (
    RepeatabilityGuidance,
    deterministic_score_order,
    sampling_weights_from_scores,
    spectrum_preserving_rerank,
)


def _entropy(probability):
    probability = probability.double()
    return -(probability * probability.log()).sum()


def _guidance(filters=3):
    return RepeatabilityGuidance(
        nn.Sequential(nn.Conv2d(1, filters, 1, bias=False))
    )


def _set_stats(guidance):
    layer, _key, filters = guidance.layers[0]
    guidance._stats = defaultdict(dict)
    contributions = [
        ([0.4, 0.1, 0.7], [0.8, 0.5, 1.0], [2, 3, 4]),
        ([0.2, 0.5, 0.2], [0.6, 0.8, 0.5], [4, 2, 3]),
        ([0.6, 0.3, 0.4], [0.9, 0.7, 0.8], [3, 5, 2]),
    ]
    for client_id, (repeat, energy, counts) in enumerate(contributions):
        count = torch.tensor(counts, dtype=torch.long)
        pairs = count.double() * (count.double() - 1.0) / 2.0
        guidance._stats[client_id][layer] = {
            "sum_r": torch.zeros((filters, 1, 1, 1), dtype=torch.float32),
            "count": count,
            "pair_dot": torch.tensor(repeat, dtype=torch.float64) * pairs,
            "energy_sum": torch.tensor(energy, dtype=torch.float64) * count,
        }


def test_probability_spectrum_sum_entropy_extrema_are_preserved():
    raw_q = {"layer": torch.tensor([0.02, 0.8, 0.3, 0.1, 0.6])}
    alternative = {"layer": torch.tensor([5.0, 1.0, 4.0, 3.0, 2.0])}
    raw_probability = sampling_weights_from_scores(raw_q, 0.75)
    reranked = spectrum_preserving_rerank(
        raw_probability, alternative, raw_q
    )
    raw = raw_probability["layer"]
    new = reranked["layer"]
    assert torch.equal(torch.sort(raw).values, torch.sort(new).values)
    assert new.sum().item() == pytest.approx(raw.sum().item(), abs=1e-15)
    assert _entropy(new).item() == pytest.approx(_entropy(raw).item(), abs=1e-15)
    assert new.max().item() == raw.max().item()
    assert new.min().item() == raw.min().item()
    assert (new.max() * new.numel()).item() == pytest.approx(
        (raw.max() * raw.numel()).item(), abs=1e-15
    )


def test_sir_changes_assignment_by_unclipped_q_minus_se_with_fixed_spectrum():
    raw_q = {"layer": torch.tensor([0.9, 0.8, 0.7, 0.6])}
    se = torch.tensor([0.6, 0.0, 0.0, 0.0])
    alternative = {"layer": raw_q["layer"].double() - se}
    raw_probability = sampling_weights_from_scores(raw_q, 0.75)
    reranked = spectrum_preserving_rerank(
        raw_probability, alternative, raw_q
    )["layer"]
    assert reranked.argmax().item() == 1
    assert torch.equal(
        torch.sort(reranked).values,
        torch.sort(raw_probability["layer"]).values,
    )


def test_sir_ties_use_raw_q_then_filter_index_deterministically():
    primary = torch.tensor([0.5, 0.5, 0.5, 0.4])
    raw_q = torch.tensor([0.8, 0.9, 0.8, 1.0])
    assert deterministic_score_order(primary, raw_q) == [1, 0, 2, 3]
    assert deterministic_score_order(primary, raw_q) == [1, 0, 2, 3]


def test_rmr_assigns_larger_raw_probability_to_larger_positive_r():
    raw_q = {"layer": torch.tensor([0.9, 0.6, 0.2])}
    positive_r = {"layer": torch.tensor([0.1, 0.5, 0.2])}
    raw_probability = sampling_weights_from_scores(raw_q, 0.75)
    reranked = spectrum_preserving_rerank(
        raw_probability, positive_r, raw_q
    )["layer"]
    assert reranked.argmax().item() == 1
    assert torch.equal(
        torch.sort(reranked).values,
        torch.sort(raw_probability["layer"]).values,
    )


def test_component_helper_raw_q_matches_original_and_exposes_r_and_e():
    original = _guidance()
    component = _guidance()
    _set_stats(original)
    _set_stats(component)
    expected = original.finalize_window()
    raw, values = component.finalize_window_with_components()
    assert torch.equal(raw["0"], expected["0"])
    assert values["0"]["repeatability"].shape == (3,)
    assert values["0"]["energy"].shape == (3,)
    assert values["0"]["eligible_clients"].tolist() == [3, 3, 3]


def test_reranking_and_component_helpers_do_not_consume_rng_state():
    guidance = _guidance()
    _set_stats(guidance)
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    raw, values = guidance.finalize_window_with_components()
    probabilities = sampling_weights_from_scores(raw, 0.75)
    spectrum_preserving_rerank(
        probabilities,
        {name: item["repeatability"].clamp_min(0.0) for name, item in values.items()},
        raw,
    )
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)
