"""Cross-Round Repeatability Diagnosis and Subspace-Selective Parameter Intervention.

This module is the numerical core of FedCRSI.  It contains no training loop
and never touches the global ``random`` / ``numpy.random`` / ``torch`` RNG
streams, so client sampling, local data shuffling and dropout remain
bit-identical to FedAvg/FedPhoenix under the same seed.

Mapping to the Method section
-----------------------------
``round_relative_deviations``   Eq. (relative_deviation):  r = Delta - mean_K(Delta)
``CrossRoundStore``             observations {r_{i,g}^t} of the current interval W
``build_pairing``               I_W, P_i and the client-balanced weights of
                                Eq. (repeatability_operator), written as
                                C_hat_g = R_g Omega R_g^T  (see below)
``filter_subspaces``            spectral decomposition of C_hat_g and the
                                index set J_g of Eq. (environment_subspace)
``diagnose_and_intervene``      Eq. (selective_intervention) applied to every
                                conv filter g, followed by the anchor update

Implementation details (Appendix level)
---------------------------------------
* Parameter components g are the output filters of every ``nn.Conv2d``
  weight, theta_g in R^{C_in*k*k}; bias, BatchNorm and Linear layers are not
  intervened on (same parameter scope as FedPhoenix's kernel reset).
* Client-balanced operator.  For client i with n_i observations
  sum_{t<t'} sym(r^t r^t'^T) = 1/2 R_i (11^T - I) R_i^T, hence
  C_hat_g = R_g Omega R_g^T with block-diagonal
  Omega_i = (11^T - I) / (2 |I_W| |P_i|),  |P_i| = n_i (n_i - 1) / 2.
* Exact low-rank spectrum.  range(C_hat_g) lies in span(R_g).  With the Gram
  matrix G = R^T R = V Lambda V^T and Z = Lambda^{1/2} V^T, the non-zero
  eigenpairs of C_hat_g are those of the n x n matrix M = Z Omega Z^T; the
  eigenvectors are U = R V Lambda^{-1/2} E.  No d x d matrix is formed and no
  approximation is made (float64 throughout).
* Finite-sample null upper spectral level.  The null must destroy the
  same-client cross-round correspondence while leaving everything else --
  the realized participation schedule, the per-round centring, the
  anisotropic noise and the exact pair structure (|P_i|, I_W) -- intact.
  Two constructions are implemented; both reduce to reweighting Omega, so
  they share the whole spectral pipeline.

  ``signflip`` (default).  Independent signs s_{i,t} in {-1,+1} per
  observation give Omega_b = (s_b s_b^T) * Omega.  Same-client cross-round
  products pick up s_{i,t} s_{i,t'} = +-1, so persistent mu_i cancels in
  expectation while *each client keeps its own deviation scale exactly*.
  Calibration needs the conditional noise distribution to be symmetric,
  which local SGD increments satisfy to good approximation.

  ``permutation``.  Independent within-round permutations of client
  identity.  Calibration additionally needs observations of different
  clients in the same round to be exchangeable.  Under Dirichlet beta=0.3
  clients hold between ~91 and ~992 samples, so their update noise scales
  differ by roughly 2.6x; same-client pairs then carry variance ~sigma_i^4
  while permuted pairs carry ~sigma_i^2 sigma_j^2, and Jensen makes the null
  systematically lighter-tailed than the statistic.  In simulation the
  false-positive rate rises from 5% (equal scales) to 20% (beta=0.3 scales)
  to 75% (6x spread), whereas the sign-flip null stays at ~5% throughout.
  It is kept only as an Appendix ablation, not for main results.

  The top eigenvalue of the reweighted operator is recorded for B draws;
  direction k of filter g is kept iff lambda_k > tau_g, the
  ((B+1)(1-alpha))-th order statistic of the null top eigenvalues.  This is
  a Monte Carlo test whose level is alpha *under its own null*; it is
  conservative across directions because every direction is compared with
  the null maximum, and tau_g is floored at 0 so only positive
  repeatability evidence is used.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


_EPS64 = float(torch.finfo(torch.float64).eps)


# --------------------------------------------------------------------------
# Parameter components and round-relative deviations
# --------------------------------------------------------------------------
def conv_filter_keys(model: nn.Module) -> list[str]:
    """State-dict keys of all convolution weights, in forward (module) order."""
    return [
        f"{name}.weight"
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]


@torch.no_grad()
def round_relative_deviations(local_states, global_state, keys):
    """r_{i,g}^t = Delta_{i,g}^t - (1/K) sum_j Delta_{j,g}^t for every filter g.

    Returns ``{key: tensor[K, C_out, C_in*k*k]}`` on the parameters' device.
    The centring uses the unweighted participant mean exactly as in the
    Method, independently of how the server aggregates.
    """
    deviations = {}
    for key in keys:
        reference = global_state[key].detach()
        delta = torch.stack(
            [state[key].detach() - reference for state in local_states]
        ).flatten(2)
        deviations[key] = delta - delta.mean(dim=0, keepdim=True)
    return deviations


class CrossRoundStore:
    """Pre-allocated buffer holding all observations of one interval.

    ``backend='ram'`` keeps the observations in host memory;
    ``backend='memmap'`` keeps them in float32 files under ``cache_dir``.
    The two backends are numerically identical: only where the bytes live
    differs.  For VGG16 one observation is ~56 MB, so an interval of
    L=20 x K=10 needs ~11 GB, which is the reason the memmap backend exists.
    """

    def __init__(self, shapes, capacity, dtype=torch.float32, backend="ram",
                 cache_dir=None):
        if backend not in {"ram", "memmap"}:
            raise ValueError("backend must be 'ram' or 'memmap'")
        self.capacity = int(capacity)
        self.backend = backend
        self._cache_dir = None
        if backend == "ram":
            self.buffers = {
                key: torch.empty((self.capacity, int(c), int(d)), dtype=dtype)
                for key, (c, d) in shapes.items()
            }
        else:
            if dtype is not torch.float32:
                raise ValueError("the memmap backend stores float32 only")
            self._cache_dir = tempfile.mkdtemp(
                prefix="crsi_", dir=cache_dir or None
            )
            self.buffers = {}
            for key, (c, d) in shapes.items():
                path = os.path.join(self._cache_dir, key.replace(".", "_") + ".f32")
                array = np.memmap(path, dtype=np.float32, mode="w+",
                                  shape=(self.capacity, int(c), int(d)))
                self.buffers[key] = array
        self.clients = np.full(self.capacity, -1, dtype=np.int64)
        self.rounds = np.full(self.capacity, -1, dtype=np.int64)
        self.size = 0

    @property
    def bytes_per_observation(self):
        total = 0
        for buffer in self.buffers.values():
            item = (buffer.dtype.itemsize if isinstance(buffer, np.memmap)
                    else buffer.element_size())
            total += int(np.prod(buffer.shape[1:])) * item
        return int(total)

    def close(self):
        if self._cache_dir is None:
            return
        for key in list(self.buffers):
            self.buffers[key] = None
        self.buffers = {}
        for name in os.listdir(self._cache_dir):
            os.remove(os.path.join(self._cache_dir, name))
        os.rmdir(self._cache_dir)
        self._cache_dir = None

    def add_round(self, round_idx, client_ids, deviations):
        client_ids = np.asarray(client_ids, dtype=np.int64)
        count = len(client_ids)
        start, stop = self.size, self.size + count
        if stop > self.capacity:
            raise RuntimeError("CrossRoundStore capacity exceeded")
        for key, value in deviations.items():
            buffer = self.buffers[key]
            if isinstance(buffer, np.memmap):
                buffer[start:stop] = value.to(device="cpu",
                                              dtype=torch.float32).numpy()
                buffer.flush()
            else:
                buffer[start:stop].copy_(value.to(device="cpu", dtype=buffer.dtype))
        self.clients[start:stop] = client_ids
        self.rounds[start:stop] = int(round_idx)
        self.size = stop

    def rows(self, key, obs_index, filter_start, filter_stop):
        """Observations of filters [start, stop) as tensor[n, c, d]."""
        buffer = self.buffers[key]
        if isinstance(buffer, np.memmap):
            index = np.asarray(obs_index, dtype=np.int64)
            block = np.ascontiguousarray(buffer[index, filter_start:filter_stop])
            return torch.from_numpy(block)
        index = torch.as_tensor(obs_index, dtype=torch.long)
        return buffer[index, filter_start:filter_stop]

    def reset(self):
        self.size = 0
        self.clients.fill(-1)
        self.rounds.fill(-1)


# --------------------------------------------------------------------------
# Pairing structure (I_W, P_i, client-balanced weights) and the null
# --------------------------------------------------------------------------
@dataclass
class PairingStructure:
    obs_index: np.ndarray  # rows of the store used by the diagnosis
    labels: np.ndarray  # client id of every used observation
    rounds: np.ndarray  # round of every used observation
    weight: np.ndarray  # Omega, float64 [n, n]
    num_clients: int  # |I_W|
    num_pairs: int  # sum_i |P_i|
    pair_counts: dict = field(default_factory=dict)

    @property
    def num_obs(self):
        return int(len(self.obs_index))


def build_pairing(clients, rounds):
    """Build I_W, P_i and Omega from the (client, round) of each observation.

    Clients observed once contribute no cross-round pair and are dropped.
    Returns ``None`` when no client was observed at least twice.
    """
    clients = np.asarray(clients, dtype=np.int64)
    rounds = np.asarray(rounds, dtype=np.int64)
    unique, counts = np.unique(clients, return_counts=True)
    repeated = unique[counts >= 2]
    if len(repeated) == 0:
        return None

    mask = np.isin(clients, repeated)
    candidate = np.flatnonzero(mask)
    order = np.lexsort((clients[candidate], rounds[candidate]))
    obs_index = candidate[order]
    labels = clients[obs_index]
    obs_rounds = rounds[obs_index]

    n = len(obs_index)
    num_clients = int(len(repeated))
    weight = np.zeros((n, n), dtype=np.float64)
    pair_counts = {}
    num_pairs = 0
    for client in repeated:
        positions = np.flatnonzero(labels == client)
        n_i = len(positions)
        if len(np.unique(obs_rounds[positions])) != n_i:
            raise ValueError("A client appears twice in the same round")
        pairs = n_i * (n_i - 1) // 2
        value = 1.0 / (2.0 * num_clients * pairs)
        weight[np.ix_(positions, positions)] = value
        weight[positions, positions] = 0.0
        pair_counts[int(client)] = int(pairs)
        num_pairs += pairs
    return PairingStructure(
        obs_index=obs_index,
        labels=labels,
        rounds=obs_rounds,
        weight=weight,
        num_clients=num_clients,
        num_pairs=int(num_pairs),
        pair_counts=pair_counts,
    )


def within_round_permutations(rounds, num_permutations, seed):
    """Independent client-identity permutations inside every round.

    ``perm[b, a]`` is the observation placed at position ``a`` in null draw
    ``b``.  Positions only exchange observations of the same round, so every
    client keeps the set of rounds in which it was observed.  A private
    ``numpy.random.Generator`` is used; the global RNG is untouched.
    """
    rounds = np.asarray(rounds)
    n = len(rounds)
    rng = np.random.default_rng(int(seed))
    groups = [np.flatnonzero(rounds == r) for r in np.unique(rounds)]
    perms = np.tile(np.arange(n, dtype=np.int64), (int(num_permutations), 1))
    for b in range(int(num_permutations)):
        for group in groups:
            if len(group) > 1:
                perms[b, group] = group[rng.permutation(len(group))]
    return perms


def permuted_weights(weight, perms):
    """Omega_pi = P Omega P^T for every null draw (shared by all filters).

    Placing observation perms[b, a] at position a is R_pi = R P, hence
    R_pi Omega R_pi^T = R (P Omega P^T) R^T:  Omega_pi[pi(a), pi(c)] = Omega[a, c].
    """
    weight = torch.as_tensor(weight, dtype=torch.float64)
    perms = torch.as_tensor(perms, dtype=torch.long)
    num_perm, n = perms.shape
    out = torch.empty((num_perm, n, n), dtype=torch.float64)
    arange = torch.arange(n)
    for b in range(num_perm):
        inverse = torch.empty(n, dtype=torch.long)
        inverse[perms[b]] = arange
        out[b] = weight[inverse][:, inverse]
    return out


def signflip_weights(weight, num_draws, seed):
    """Omega_s = (s s^T) * Omega for independent per-observation signs.

    Sign-flipping observation (i, t) maps R -> diag(s) R, so
    (diag(s) R)^T Omega (diag(s) R) = R^T ((s s^T) * Omega) R.  Each client's
    own deviation scale is preserved exactly, which is what keeps the test
    calibrated under client-heterogeneous update noise.
    """
    weight = torch.as_tensor(weight, dtype=torch.float64)
    n = weight.shape[0]
    rng = np.random.default_rng(int(seed))
    signs = torch.as_tensor(
        rng.integers(0, 2, size=(int(num_draws), n)) * 2.0 - 1.0, dtype=torch.float64
    )
    return signs.unsqueeze(-1) * signs.unsqueeze(-2) * weight


def null_weights(kind, weight, rounds, num_draws, seed):
    """Dispatch to the requested null construction."""
    if kind == "signflip":
        return signflip_weights(weight, num_draws, seed)
    if kind == "permutation":
        return permuted_weights(
            weight, within_round_permutations(rounds, num_draws, seed)
        )
    raise ValueError(f"unknown crsi_null '{kind}'")


def null_exceedance_allowance(num_permutations, alpha):
    """Largest number of null draws allowed to reach the observed value.

    Monte Carlo p-value: p = (1 + #{null >= obs}) / (B + 1) <= alpha.
    """
    allowance = math.floor((num_permutations + 1) * alpha + 1e-9) - 1
    if allowance < 0:
        raise ValueError(
            "crsi_null_permutations and crsi_null_alpha must satisfy "
            "(B + 1) * alpha >= 1"
        )
    if allowance >= num_permutations:
        raise ValueError("crsi_null_alpha is too large for the permutation count")
    return int(allowance)


# --------------------------------------------------------------------------
# Spectral subspace of C_hat_g (exact low-rank form) and null level
# --------------------------------------------------------------------------
@torch.no_grad()
def filter_subspaces(gram, weight, weight_perm, allowance, feature_dim):
    """Environment-sensitive subspaces for a batch of filters.

    Args:
        gram: float64 [c, n, n] Gram matrices G = R^T R of the observations.
        weight: float64 [n, n] client-balanced pairing matrix Omega.
        weight_perm: float64 [B, n, n] reweighted pairing matrices of the
            null draws (``null_weights``).
        allowance: output of ``null_exceedance_allowance``.
        feature_dim: d_g (used for the numerical rank tolerance).

    Returns a dict with
        coef: float64 [c, n, n]  A such that U_g = R_g A_g (unselected columns 0)
        eigenvalues: float64 [c, n] eigenvalues of C_hat_g (ascending)
        tau: float64 [c] null upper spectral level
        selected: bool [c, n] membership of J_g
    """
    gram = 0.5 * (gram + gram.transpose(1, 2))
    n = gram.shape[-1]
    lam_g, vec_g = torch.linalg.eigh(gram)
    top = lam_g[:, -1:].clamp_min(0.0)
    tolerance = top * (max(n, int(feature_dim)) * _EPS64)
    keep = lam_g > tolerance
    sqrt_lam = torch.where(keep, lam_g.clamp_min(0.0).sqrt(), torch.zeros_like(lam_g))
    inv_sqrt = torch.where(keep, 1.0 / sqrt_lam.clamp_min(1e-300), torch.zeros_like(lam_g))

    z = sqrt_lam.unsqueeze(-1) * vec_g.transpose(1, 2)  # [c, n(coords), n(obs)]
    m = z @ weight @ z.transpose(1, 2)
    m = 0.5 * (m + m.transpose(1, 2))
    eigenvalues, eigenvectors = torch.linalg.eigh(m)

    # Null: both constructions act on the observations as a linear
    # reweighting of Omega, so the same Z is reused for all B draws.
    c = z.shape[0]
    num_perm = weight_perm.shape[0]
    y = (z.reshape(c * n, n) @ weight_perm.permute(1, 0, 2).reshape(n, num_perm * n))
    y = y.reshape(c, n, num_perm, n).permute(0, 2, 1, 3)  # [c, B, n, n]
    m_perm = y @ z.transpose(1, 2).unsqueeze(1)
    del y
    m_perm = m_perm.add_(m_perm.transpose(-1, -2).clone()).mul_(0.5)
    null_top = torch.linalg.eigvalsh(m_perm)[..., -1]  # [c, B]
    del m_perm
    tau = torch.topk(null_top, allowance + 1, dim=1).values[:, -1]
    tau = tau.clamp_min(0.0)

    # The Monte Carlo p-value counts ties as exceedances (#{null >= obs}).
    # Degenerate draws (e.g. an all-identity permutation when no round holds
    # two repeated clients) reproduce the observed spectrum up to rounding,
    # so ties are resolved with a relative margin far below any
    # statistically meaningful gap.
    scale = torch.maximum(tau, eigenvalues.abs().amax(dim=-1))
    margin = 1e-9 * scale
    selected = eigenvalues > (tau + margin).unsqueeze(-1)
    coef = (vec_g * inv_sqrt.unsqueeze(1)) @ (
        eigenvectors * selected.unsqueeze(1).to(eigenvectors.dtype)
    )
    return {
        "coef": coef,
        "eigenvalues": eigenvalues,
        "tau": tau,
        "selected": selected,
        "null_top": null_top,
    }


def _chunk_ranges(total, size):
    size = max(1, int(size))
    return [(start, min(total, start + size)) for start in range(0, total, size)]


@torch.no_grad()
def diagnose_and_intervene(
    global_state,
    anchor,
    store,
    keys,
    *,
    num_draws,
    alpha,
    seed,
    device,
    null="signflip",
    apply=True,
    chunk_bytes=1 << 30,
    workers=1,
):
    """Cross-Round Repeatability Diagnosis + Subspace-Selective Intervention.

    ``global_state`` holds the standard aggregate theta_tilde^{t+1}; its conv
    weights are replaced in place by
        theta_g <- theta_tilde_g - P_g (theta_tilde_g - a_g)
    when ``apply`` is True.  Returns a JSON-serialisable diagnostic record.

    Every observation is read exactly once: each filter chunk is loaded, its
    Gram matrices, spectral subspaces and correction are computed, and the
    chunk is released.  This matters for the memmap backend, where a second
    pass would double the disk traffic.
    """
    started = time.perf_counter()
    record = {
        "observations": int(store.size),
        "applied": bool(apply),
        "null": str(null),
        "layers": [],
    }
    pairing = build_pairing(store.clients[: store.size], store.rounds[: store.size])
    if pairing is None:
        record.update(
            {"repeat_clients": 0, "used_observations": 0, "pairs": 0,
             "filters": 0, "filters_with_subspace": 0,
             "drift_sq": 0.0, "removed_sq": 0.0, "removed_fraction": 0.0,
             "seconds": time.perf_counter() - started}
        )
        return record

    allowance = null_exceedance_allowance(num_draws, alpha)
    n = pairing.num_obs
    weight = torch.as_tensor(pairing.weight, dtype=torch.float64)
    weight_null = null_weights(null, weight, pairing.rounds, num_draws, seed)
    record.update(
        {
            "repeat_clients": pairing.num_clients,
            "used_observations": n,
            "pairs": pairing.num_pairs,
            "null_draws": int(num_draws),
            "alpha": float(alpha),
        }
    )

    total_filters = total_selected = 0
    total_drift = total_removed = 0.0
    for key in keys:
        param = global_state[key]
        c_out = int(param.shape[0])
        d = int(param[0].numel())
        flat = param.detach().reshape(c_out, d)
        drift = flat.to(torch.float64) - anchor[key].detach().reshape(c_out, d).to(
            torch.float64
        )
        # A chunk holds x (float64) plus, per filter, the null tensors: y and
        # m_perm are live together and symmetrisation needs one more copy, so
        # budget ~3B matrices of size n x n per filter.
        per_filter = 8 * (n * d + n * n * (3 * num_draws + 6))
        chunk = max(1, int(chunk_bytes) // max(1, per_filter))
        ranks = torch.zeros(c_out, dtype=torch.long)
        removed = torch.zeros(c_out, dtype=torch.float64)
        top_ratio = torch.zeros(c_out, dtype=torch.float64)

        for start, stop in _chunk_ranges(c_out, chunk):
            x = store.rows(key, pairing.obs_index, start, stop)
            x = x.to(device=device, dtype=torch.float64).transpose(0, 1)  # [c, n, d]
            gram = (x @ x.transpose(1, 2)).cpu()
            proj = torch.einsum("cnd,cd->cn", x, drift[start:stop].to(device)).cpu()

            width = stop - start
            coeff = torch.zeros((width, n), dtype=torch.float64)
            sub = max(1, width // max(1, int(workers)))

            def solve(bounds):
                lo, hi = bounds
                out = filter_subspaces(
                    gram[lo:hi], weight, weight_null, allowance, d
                )
                a = out["coef"]
                # U^T d = A^T (R^T d);  P d = R A (A^T R^T d)
                coord = torch.einsum("cnk,cn->ck", a, proj[lo:hi])
                coeff[lo:hi] = torch.einsum("cnk,ck->cn", a, coord)
                ranks[start + lo:start + hi] = out["selected"].sum(dim=1)
                removed[start + lo:start + hi] = (coord ** 2).sum(dim=1)
                tau = out["tau"]
                top_ratio[start + lo:start + hi] = torch.where(
                    tau > 0,
                    out["eigenvalues"][:, -1] / tau.clamp_min(1e-300),
                    torch.zeros_like(tau),
                )

            bounds = _chunk_ranges(width, sub)
            if workers > 1 and len(bounds) > 1:
                with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                    list(pool.map(solve, bounds))
            else:
                for item in bounds:
                    solve(item)

            if apply and bool((ranks[start:stop] > 0).any()):
                correction = torch.einsum("cnd,cn->cd", x, coeff.to(device))
                updated = flat[start:stop].to(torch.float64) - correction.to(flat.device)
                flat[start:stop] = updated.to(param.dtype)
            del x, gram, proj

        drift_sq = float((drift ** 2).sum())
        removed_sq = float(removed.sum())
        layer_selected = int((ranks > 0).sum())
        total_filters += c_out
        total_selected += layer_selected
        total_drift += drift_sq
        total_removed += removed_sq
        record["layers"].append(
            {
                "key": key,
                "filters": c_out,
                "dim": d,
                "filters_with_subspace": layer_selected,
                "mean_rank": float(ranks.double().mean()),
                "max_rank": int(ranks.max()),
                "drift_sq": drift_sq,
                "removed_sq": removed_sq,
                "removed_fraction": removed_sq / drift_sq if drift_sq > 0 else 0.0,
                "median_top_over_tau": float(top_ratio.median()),
            }
        )

    record.update(
        {
            "filters": total_filters,
            "filters_with_subspace": total_selected,
            "drift_sq": total_drift,
            "removed_sq": total_removed,
            "removed_fraction": total_removed / total_drift if total_drift > 0 else 0.0,
            "seconds": time.perf_counter() - started,
        }
    )
    return record


def partition_sha256(path):
    """SHA-256 of the client-partition file, recorded for reproducibility."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
