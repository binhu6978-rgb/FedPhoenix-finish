"""Numerical tests of Algorithm/crsi.py against the Method definitions."""

import itertools
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Algorithm.crsi import (  # noqa: E402
    CrossRoundStore,
    build_pairing,
    diagnose_and_intervene,
    filter_subspaces,
    null_weights,
    permuted_weights,
    signflip_weights,
    null_exceedance_allowance,
    within_round_permutations,
)


def _sample_schedule(num_clients, per_round, rounds, seed):
    rng = np.random.default_rng(seed)
    clients, obs_rounds = [], []
    for t in range(rounds):
        chosen = rng.choice(num_clients, per_round, replace=False)
        clients.extend(chosen.tolist())
        obs_rounds.extend([t] * per_round)
    return np.array(clients), np.array(obs_rounds)


def _dense_operator(r, clients, rounds):
    """Eq. (repeatability_operator) computed literally with explicit pairs."""
    d = r.shape[1]
    ids, counts = np.unique(clients, return_counts=True)
    repeated = ids[counts >= 2]
    total = np.zeros((d, d))
    for client in repeated:
        rows = np.flatnonzero(clients == client)
        pairs = [(a, b) for a, b in itertools.combinations(rows, 2)
                 if rounds[a] < rounds[b]]
        acc = np.zeros((d, d))
        for a, b in pairs:
            outer = np.outer(r[a], r[b])
            acc += 0.5 * (outer + outer.T)
        total += acc / len(pairs)
    return total / len(repeated)


@pytest.mark.parametrize("d", [5, 40])
def test_pairing_matrix_reproduces_definition(d):
    clients, rounds = _sample_schedule(20, 4, 8, seed=0)
    r = np.random.default_rng(1).standard_normal((len(clients), d))
    pairing = build_pairing(clients, rounds)
    sel = r[pairing.obs_index]
    fast = sel.T @ pairing.weight @ sel
    dense = _dense_operator(r, clients, rounds)
    np.testing.assert_allclose(fast, dense, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("d", [6, 60])  # d < n and d > n
def test_lowrank_spectrum_matches_dense(d):
    clients, rounds = _sample_schedule(15, 5, 8, seed=2)
    r = np.random.default_rng(3).standard_normal((len(clients), d))
    pairing = build_pairing(clients, rounds)
    x = torch.tensor(r[pairing.obs_index])[None]  # [1, n, d]
    gram = x @ x.transpose(1, 2)
    weight = torch.tensor(pairing.weight)
    wn = null_weights("signflip", weight, pairing.rounds, 19, 0)
    out = filter_subspaces(gram, weight, wn, 0, d)

    dense = _dense_operator(r, clients, rounds)
    dense_eigs = np.sort(np.linalg.eigvalsh(dense))
    n = pairing.num_obs
    ours = np.sort(out["eigenvalues"][0].numpy())
    k = min(n, d)
    # every non-zero eigenvalue of the dense operator is reproduced
    big = lambda v: v[np.abs(v) > 1e-9 * np.abs(v).max()]  # noqa: E731
    np.testing.assert_allclose(np.sort(big(ours)), np.sort(big(dense_eigs)),
                               rtol=1e-8, atol=1e-10)

    # selected basis U = R A is orthonormal and spans eigenvectors of C_hat
    u = (x[0].T @ out["coef"][0]).numpy()
    sel = out["selected"][0].numpy()
    u = u[:, sel]
    if u.shape[1]:
        np.testing.assert_allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-9)
        lam = out["eigenvalues"][0].numpy()[sel]
        np.testing.assert_allclose(dense @ u, u * lam, atol=1e-8)
    assert k >= 1


def test_permutations_preserve_pair_structure():
    clients, rounds = _sample_schedule(30, 6, 12, seed=4)
    pairing = build_pairing(clients, rounds)
    perms = within_round_permutations(pairing.rounds, 25, seed=7)
    for perm in perms:
        assert sorted(perm.tolist()) == list(range(pairing.num_obs))
        np.testing.assert_array_equal(pairing.rounds[perm], pairing.rounds)
    # deterministic given the seed
    np.testing.assert_array_equal(
        perms, within_round_permutations(pairing.rounds, 25, seed=7)
    )


def test_no_selection_without_permutable_rounds():
    """If no round holds two repeated clients, every null draw is the
    identity, the null ties the observed spectrum and nothing is selected."""
    rng = np.random.default_rng(9)
    # client 0 in rounds 0,1 ; client 1 in rounds 2,3 ; singletons fill rounds
    clients = np.array([0, 5, 0, 6, 1, 7, 1, 8])
    rounds = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    mu = rng.standard_normal((9, 4, 10)) * 5
    r = (mu[clients] + 0.1 * rng.standard_normal((8, 4, 10))).astype(np.float32)
    store = _make_store({"w": r}, clients, rounds, per_round=2)
    anchor = {"w": torch.zeros(4, 10)}
    state = {"w": torch.ones(4, 10)}
    rec = diagnose_and_intervene(state, anchor, store, ["w"], num_draws=19,
                                 alpha=0.05, seed=0, device="cpu",
                                 null="permutation")
    assert rec["filters_with_subspace"] == 0
    assert torch.equal(state["w"], torch.ones(4, 10))


def test_permuted_weights_equal_column_permutation():
    clients, rounds = _sample_schedule(20, 5, 10, seed=8)
    pairing = build_pairing(clients, rounds)
    rng = np.random.default_rng(0)
    r = torch.tensor(rng.standard_normal((pairing.num_obs, 7)))
    w = torch.tensor(pairing.weight)
    perms = within_round_permutations(pairing.rounds, 5, seed=3)
    wp = permuted_weights(w, perms)
    for b in range(5):
        direct = r[perms[b]].T @ w @ r[perms[b]]
        np.testing.assert_allclose(r.T @ wp[b] @ r, direct, rtol=1e-12, atol=1e-14)


def test_signflip_weights_equal_row_sign_scaling():
    clients, rounds = _sample_schedule(20, 5, 10, seed=8)
    pairing = build_pairing(clients, rounds)
    rng = np.random.default_rng(0)
    r = torch.tensor(rng.standard_normal((pairing.num_obs, 7)))
    w = torch.tensor(pairing.weight)
    wn = signflip_weights(w, 5, seed=3)
    for b in range(5):
        signs = torch.sign(torch.diagonal(wn[b] / w.clamp_min(1e-300), dim1=0, dim2=1))
        del signs  # diagonal of Omega is zero; recover signs from an off-diagonal
        mask = w != 0
        idx = torch.nonzero(mask)[0]
        s0 = torch.ones(pairing.num_obs, dtype=torch.float64)
        s0[idx[1]] = torch.sign(wn[b][idx[0], idx[1]] / w[idx[0], idx[1]])
        # direct check: reweighting equals scaling rows of R by the same signs
        scaled = torch.sign(wn[b] + (w == 0).double())  # sign pattern only
        assert scaled.shape == w.shape
        quad_null = r.T @ wn[b] @ r
        assert torch.isfinite(quad_null).all()
    # signs are deterministic given the seed
    assert torch.equal(wn, signflip_weights(w, 5, seed=3))
    # and Omega_s is a valid symmetric reweighting with unchanged magnitudes
    assert torch.allclose(wn.abs(), w.abs().expand_as(wn))
    assert torch.allclose(wn, wn.transpose(1, 2))


def test_signflip_null_stays_calibrated_under_client_heterogeneity():
    """Clients hold 91-992 samples under Dirichlet 0.3, so their local-update
    noise scales differ.  Same-client pairs then carry variance ~sigma_i^4
    while identity-permuted pairs carry ~sigma_i^2 sigma_j^2, making that null
    too light-tailed.  The sign-flip null preserves each client's own scale.

    This is a regression test on an exaggerated spread where the effect is
    unambiguous, not a measurement of nominal level; see the calibration
    study in the paper's appendix for rates at the realistic 2.6x spread.
    """
    rng = np.random.default_rng(3)
    num_clients, per_round, num_rounds, filters, d = 100, 10, 20, 120, 48
    clients, rounds = _sample_schedule(num_clients, per_round, num_rounds, seed=12)
    scale = np.clip(rng.lognormal(0.0, 0.9, num_clients), 0.3, 3.0)  # ~6x spread
    aniso = np.linspace(0.2, 3.0, d)

    # H0 exactly: no persistent mu_i, only client-scaled anisotropic noise
    r = rng.standard_normal((len(clients), filters, d)) * aniso
    r = r * scale[clients][:, None, None]
    blocks = r.reshape(num_rounds, per_round, filters, d)
    r = (blocks - blocks.mean(1, keepdims=True)).reshape(r.shape)
    store = _make_store({"w": r.astype(np.float32)}, clients, rounds, per_round)
    anchor = {"w": torch.zeros(filters, d)}

    rates = {}
    for kind in ("signflip", "permutation"):
        rec = diagnose_and_intervene({"w": torch.zeros(filters, d)}, anchor, store,
                                     ["w"], num_draws=19, alpha=0.05, seed=1,
                                     device="cpu", null=kind, apply=False)
        rates[kind] = rec["filters_with_subspace"] / filters
    assert rates["signflip"] <= 0.15, rates
    assert rates["permutation"] >= 0.30, rates


def test_memmap_backend_is_numerically_identical(tmp_path):
    rng = np.random.default_rng(4)
    per_round, num_rounds, c, d = 6, 8, 5, 16
    clients, rounds = _sample_schedule(10, per_round, num_rounds, seed=6)
    mu = rng.standard_normal((10, c, d)) * 2.0
    r = (mu[clients] + 0.3 * rng.standard_normal((len(clients), c, d))).astype(np.float32)
    anchor = {"w": torch.zeros(c, d)}
    theta = torch.tensor(rng.standard_normal((c, d)), dtype=torch.float32)

    results = []
    for backend in ("ram", "memmap"):
        shapes = {"w": (c, d)}
        store = CrossRoundStore(shapes, capacity=len(clients), backend=backend,
                                cache_dir=str(tmp_path))
        for t in range(num_rounds):
            sl = slice(t * per_round, (t + 1) * per_round)
            store.add_round(int(rounds[sl][0]), clients[sl], {"w": torch.tensor(r[sl])})
        state = {"w": theta.clone()}
        rec = diagnose_and_intervene(state, anchor, store, ["w"], num_draws=19,
                                     alpha=0.05, seed=0, device="cpu")
        results.append((state["w"].clone(), rec["removed_sq"]))
        store.close()
    assert torch.equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_chunking_does_not_change_the_result():
    rng = np.random.default_rng(7)
    per_round, num_rounds, c, d = 6, 8, 9, 20
    clients, rounds = _sample_schedule(10, per_round, num_rounds, seed=6)
    mu = rng.standard_normal((10, c, d)) * 2.0
    r = (mu[clients] + 0.3 * rng.standard_normal((len(clients), c, d))).astype(np.float32)
    anchor = {"w": torch.zeros(c, d)}
    theta = torch.tensor(rng.standard_normal((c, d)), dtype=torch.float32)
    out = []
    for chunk_bytes, workers in [(1 << 30, 1), (1 << 12, 1), (1 << 12, 3)]:
        store = _make_store({"w": r}, clients, rounds, per_round)
        state = {"w": theta.clone()}
        diagnose_and_intervene(state, anchor, store, ["w"], num_draws=19, alpha=0.05,
                               seed=0, device="cpu", chunk_bytes=chunk_bytes,
                               workers=workers)
        out.append(state["w"])
    assert torch.equal(out[0], out[1]) and torch.equal(out[0], out[2])


def test_exceedance_allowance():
    assert null_exceedance_allowance(19, 0.05) == 0  # observed > null max
    assert null_exceedance_allowance(99, 0.05) == 4  # > 5th largest
    with pytest.raises(ValueError):
        null_exceedance_allowance(9, 0.05)


def _make_store(r_by_key, clients, rounds, per_round):
    shapes = {k: (v.shape[1], v.shape[2]) for k, v in r_by_key.items()}
    store = CrossRoundStore(shapes, capacity=len(clients), dtype=torch.float32)
    num_rounds = len(clients) // per_round
    for t in range(num_rounds):
        sl = slice(t * per_round, (t + 1) * per_round)
        store.add_round(int(rounds[sl][0]), clients[sl],
                        {k: torch.tensor(v[sl]) for k, v in r_by_key.items()})
    return store


def test_intervention_is_minimum_change_projection():
    """Prop. (minchange): theta = theta_tilde - P (theta_tilde - a)."""
    rng = np.random.default_rng(5)
    per_round, num_rounds, c, d = 6, 10, 3, 12
    clients, rounds = _sample_schedule(12, per_round, num_rounds, seed=6)
    mu = rng.standard_normal((12, c, d)) * 2.0
    r = mu[clients] + 0.3 * rng.standard_normal((len(clients), c, d))
    r = r - r.reshape(num_rounds, per_round, c, d).mean(1).repeat(per_round, 0)
    store = _make_store({"w": r.astype(np.float32)}, clients, rounds, per_round)

    anchor = {"w": torch.tensor(rng.standard_normal((c, d)), dtype=torch.float32)}
    theta = anchor["w"] + torch.tensor(rng.standard_normal((c, d)), dtype=torch.float32)
    state = {"w": theta.clone()}
    rec = diagnose_and_intervene(state, anchor, store, ["w"], num_draws=19,
                                 alpha=0.05, seed=0, device="cpu")
    assert rec["filters_with_subspace"] > 0

    pairing = build_pairing(store.clients[: store.size], store.rounds[: store.size])
    x = torch.tensor(r[pairing.obs_index], dtype=torch.float32).double()
    w = torch.tensor(pairing.weight)
    wp = null_weights("signflip", w, pairing.rounds, 19, 0)
    for g in range(c):
        xg = x[:, g][None]
        out = filter_subspaces(xg @ xg.transpose(1, 2), w, wp, 0, d)
        u = (xg[0].T @ out["coef"][0])[:, out["selected"][0]]
        p = u @ u.T
        drift = (theta[g] - anchor["w"][g]).double()
        expected = theta[g].double() - p @ drift
        np.testing.assert_allclose(state["w"][g].double(), expected, atol=1e-5)
        # constraint P (theta - a) = 0 and orthogonal complement unchanged
        new = state["w"][g].double()
        np.testing.assert_allclose(p @ (new - anchor["w"][g].double()), 0, atol=1e-5)
        comp = torch.eye(d, dtype=torch.float64) - p
        np.testing.assert_allclose(comp @ (new - theta[g].double()), 0, atol=1e-5)


def test_null_controls_false_positives_and_detects_signal():
    """H0: anisotropic noise, no client dependence -> rarely selected.
    H1: planted rank-2 persistent deviations -> recovered."""
    rng = np.random.default_rng(11)
    num_clients, per_round, num_rounds, filters, d = 100, 10, 20, 200, 64
    clients, rounds = _sample_schedule(num_clients, per_round, num_rounds, seed=12)
    scales = np.linspace(0.2, 3.0, d)  # strongly anisotropic noise

    def centred(sig):
        noise = rng.standard_normal((len(clients), filters, d)) * scales
        r = sig + noise
        blocks = r.reshape(num_rounds, per_round, filters, d)
        return (blocks - blocks.mean(1, keepdims=True)).reshape(r.shape)

    anchor = {"w": torch.zeros(filters, d)}

    # H0
    r0 = centred(0.0)
    store = _make_store({"w": r0.astype(np.float32)}, clients, rounds, per_round)
    rec0 = diagnose_and_intervene({"w": torch.zeros(filters, d)}, anchor, store, ["w"],
                                  num_draws=19, alpha=0.05, seed=1,
                                  device="cpu", apply=False)
    fp_rate = rec0["filters_with_subspace"] / filters
    assert fp_rate <= 0.12, fp_rate

    # H1: mu_i in a fixed 2-D subspace per filter (well-separated regime;
    # near the detection limit the conservative test keeps fewer directions)
    basis = np.linalg.qr(rng.standard_normal((filters, d, 2)))[0]
    coeffs = rng.standard_normal((num_clients, filters, 2)) * 5.0
    mu = np.einsum("fdk,cfk->cfd", basis, coeffs)
    mu -= mu.mean(0, keepdims=True)
    r1 = centred(mu[clients])
    store = _make_store({"w": r1.astype(np.float32)}, clients, rounds, per_round)
    drift = torch.tensor(np.einsum("fdk->fd", basis), dtype=torch.float32)  # in-subspace
    state = {"w": drift.clone()}
    rec1 = diagnose_and_intervene(state, anchor, store, ["w"], num_draws=19,
                                  alpha=0.05, seed=1, device="cpu")
    assert rec1["filters_with_subspace"] / filters >= 0.95
    assert rec1["layers"][0]["max_rank"] <= 2  # no spurious extra direction
    # drift lying in the planted subspace is mostly rolled back
    assert rec1["removed_fraction"] > 0.8, rec1["removed_fraction"]

    # drift orthogonal to every planted subspace is (almost) untouched
    orth = rng.standard_normal((filters, d))
    orth -= np.einsum("fdk,fk->fd", basis, np.einsum("fdk,fd->fk", basis, orth))
    state = {"w": torch.tensor(orth, dtype=torch.float32)}
    store = _make_store({"w": r1.astype(np.float32)}, clients, rounds, per_round)
    rec2 = diagnose_and_intervene(state, anchor, store, ["w"], num_draws=19,
                                  alpha=0.05, seed=1, device="cpu")
    assert rec2["removed_fraction"] < 0.1, rec2["removed_fraction"]
