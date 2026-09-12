"""FedCRSI: Cross-Round Repeatability Diagnosis + Subspace-Selective Intervention.

The round loop is the repository's FedAvg loop (``main_fed.FedAvg``) with the
same calls in the same order: ``np.random.choice`` client sampling,
``LocalUpdate_FedAvg`` local training from the current global model,
sample-size weighted ``Aggregation`` and full-test-set evaluation.  The method
adds two hooks that consume no global RNG:

  hook A (every round)     store r_{i,g}^t of the participating clients
  hook B (interval end)    diagnose U_g for every conv filter and roll back
                           P_g (theta_tilde_g - a_g); reset anchor and store

Consequently, with ``--crsi_dry_run 1`` the global trajectory is identical to
FedAvg, and with interventions enabled it is identical to FedAvg up to the end
of the first diagnosis interval.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import time

import numpy as np
import torch

from Algorithm.crsi import (
    CrossRoundStore,
    conv_filter_keys,
    diagnose_and_intervene,
    partition_sha256,
    round_relative_deviations,
)
from models.Fed import Aggregation
from models.test import evaluate_round_accuracy, print_peak_accuracy
from models.Update import LocalUpdate_FedAvg


METHOD_NAME = "FedCRSI"
_DTYPES = {"float32": torch.float32, "float16": torch.float16}


def _run_stem(args):
    suffix = ""
    if getattr(args, "run_name", ""):
        safe = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(args.run_name)
        )
        suffix = f"_{safe}" if safe else ""
    return f"{args.dataset}_{args.model}_{args.algorithm}_seed{args.seed}{suffix}"


def _validate(args, clients_per_round):
    if args.crsi_interval < 2:
        raise ValueError("crsi_interval must be >= 2 (cross-round pairs need two rounds)")
    if not 0.0 < args.crsi_null_alpha < 1.0:
        raise ValueError("crsi_null_alpha must be in (0, 1)")
    if (args.crsi_null_draws + 1) * args.crsi_null_alpha < 1.0 - 1e-9:
        raise ValueError("need (crsi_null_draws + 1) * crsi_null_alpha >= 1")
    if args.crsi_store_dtype not in _DTYPES:
        raise ValueError(f"crsi_store_dtype must be one of {sorted(_DTYPES)}")
    if args.crsi_store_dtype == "float16":
        print(
            "WARNING: crsi_store_dtype=float16 quantises the observations that "
            "the spectrum and the selection threshold are computed from, so it "
            "is an approximate variant, not a memory optimisation of the same "
            "algorithm. Use float32 (with crsi_store_backend=memmap if host "
            "memory is tight) for reported results.",
            flush=True,
        )
    if args.crsi_store_backend == "memmap" and args.crsi_store_dtype != "float32":
        raise ValueError("the memmap backend stores float32 only")
    if args.crsi_null not in {"signflip", "permutation"}:
        raise ValueError("crsi_null must be 'signflip' or 'permutation'")
    if args.crsi_null == "permutation":
        print(
            "WARNING: crsi_null=permutation assumes that observations of "
            "different clients in the same round are exchangeable. Under "
            "Dirichlet heterogeneity their update scales differ and the test "
            "becomes anti-conservative; this setting is an ablation.",
            flush=True,
        )
    if clients_per_round < 2:
        raise ValueError("round-relative deviations need at least two participants")


def _cpu_workers(args):
    if args.crsi_workers > 0:
        return int(args.crsi_workers)
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def FedCRSI(args, net_glob, dataset_train, dataset_test, dict_users):
    m = max(int(args.frac * args.num_users), 1)
    _validate(args, m)
    net_glob.train()

    keys = conv_filter_keys(net_glob)
    state0 = net_glob.state_dict()
    shapes = {k: (state0[k].shape[0], state0[k][0].numel()) for k in keys}
    store = CrossRoundStore(
        shapes,
        capacity=args.crsi_interval * m,
        dtype=_DTYPES[args.crsi_store_dtype],
        backend=args.crsi_store_backend,
        cache_dir=args.crsi_cache_dir or None,
    )
    anchor = {k: state0[k].detach().clone() for k in keys}
    workers = _cpu_workers(args)

    os.makedirs(args.crsi_log_dir, exist_ok=True)
    os.makedirs(args.metrics_log_dir, exist_ok=True)
    stem = _run_stem(args)
    diag_path = os.path.abspath(os.path.join(args.crsi_log_dir, f"{stem}_interventions.jsonl"))
    open(diag_path, "w", encoding="utf-8").close()
    per_obs_mb = store.bytes_per_observation / 2**20
    print(
        f"FedCRSI: {len(keys)} conv layers, {sum(c for c, _ in shapes.values())} filters, "
        f"interval={args.crsi_interval}, null={args.crsi_null} "
        f"B={args.crsi_null_draws} alpha={args.crsi_null_alpha}, "
        f"store={store.capacity} obs x {per_obs_mb:.1f} MB "
        f"({store.capacity * per_obs_mb / 1024:.1f} GB, {store.backend}), "
        f"workers={workers}, dry_run={bool(args.crsi_dry_run)}",
        flush=True,
    )

    # Record the exact client split so that every method can be shown to have
    # used the same one; --generate_data 0 means this file is authoritative.
    partition_file = os.path.join(
        "data",
        f"{args.dataset}_{args.num_users}"
        + ("_iid" if args.iid else f"_noniidCase{args.noniid_case}")
        + (f"_beta{args.data_beta}" if args.noniid_case > 4 else "")
        + ".json",
    )
    provenance = {"partition_path": os.path.abspath(partition_file)}
    if os.path.isfile(partition_file):
        provenance["partition_sha256"] = partition_sha256(partition_file)
    else:
        provenance["partition_sha256"] = None
    provenance["client_sizes"] = [len(dict_users[i]) for i in sorted(dict_users)]
    print(
        f"PARTITION path={provenance['partition_path']} "
        f"sha256={provenance['partition_sha256']}",
        flush=True,
    )

    acc = []
    rows = []
    interval_idx = 0
    for iter in range(args.epochs):
        round_start = time.perf_counter()
        print('*' * 80)
        print('Round {:3d}'.format(iter))

        # ---- identical to main_fed.FedAvg --------------------------------
        w_locals = []
        lens = []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            net_local = None
            net_local = copy.deepcopy(net_glob).to(args.device)
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train,
                                       idxs=dict_users[idx], dataset_test=dataset_test)
            w = local.train(net=net_local)
            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))

        # ---- hook A: round-relative deviations r_{i,g}^t ------------------
        store.add_round(
            iter, idxs_users,
            round_relative_deviations(w_locals, net_glob.state_dict(), keys),
        )

        w_glob = Aggregation(w_locals, lens)  # theta_tilde^{t+1}, unchanged
        del w_locals

        # ---- hook B: diagnosis + selective intervention at interval end ---
        diag_seconds = 0.0
        intervened = False
        if (iter + 1) % args.crsi_interval == 0:
            record = diagnose_and_intervene(
                w_glob, anchor, store, keys,
                num_draws=args.crsi_null_draws,
                alpha=args.crsi_null_alpha,
                seed=int(args.seed) + 300000 + interval_idx * 100003,
                device=args.device,
                null=args.crsi_null,
                apply=not bool(args.crsi_dry_run),
                chunk_bytes=int(args.crsi_chunk_mb) << 20,
                workers=workers,
            )
            record.update({"interval": interval_idx, "round": iter + 1,
                           "start_round": iter + 2 - args.crsi_interval})
            diag_seconds = record["seconds"]
            intervened = record["applied"] and record["filters_with_subspace"] > 0
            with open(diag_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"CRSI_INTERVENTION round={iter + 1} repeat_clients={record['repeat_clients']} "
                f"pairs={record['pairs']} filters_with_subspace="
                f"{record['filters_with_subspace']}/{record['filters']} "
                f"removed_fraction={record['removed_fraction']:.4f} "
                f"applied={int(record['applied'])} seconds={diag_seconds:.1f}",
                flush=True,
            )
            anchor = {k: w_glob[k].detach().clone() for k in keys}
            store.reset()
            interval_idx += 1

        net_glob.load_state_dict(w_glob)
        item_acc = evaluate_round_accuracy(net_glob, dataset_test, args, iter + 1)
        acc.append(item_acc)
        rows.append({
            "round": iter + 1,
            "algorithm": args.algorithm,
            "seed": int(args.seed),
            "test_accuracy": item_acc,
            "intervened": int(intervened),
            "round_seconds": time.perf_counter() - round_start,
            "diagnosis_seconds": diag_seconds,
            "selected_clients": json.dumps([int(c) for c in idxs_users.tolist()]),
        })

    print_peak_accuracy(acc, args.algorithm)
    csv_path = os.path.abspath(os.path.join(args.metrics_log_dir, f"{stem}.csv"))
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.metrics_log_dir, f"{stem}_config.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"args": vars(args), "provenance": provenance}, handle,
                  ensure_ascii=False, indent=2, default=str)
    store.close()
    print(f"Unified training metrics saved to {csv_path}")
    print(f"FedCRSI diagnostics saved to {diag_path}")
    return acc
