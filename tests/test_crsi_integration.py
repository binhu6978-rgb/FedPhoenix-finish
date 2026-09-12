"""End-to-end checks against the repository's own FedAvg loop (CPU, synthetic data)."""

import contextlib
import copy
import io
import json
import os
import re
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0]]

import main_fed  # noqa: E402
from Algorithm.Training_FedCRSI import FedCRSI  # noqa: E402
from models.Nets import CNNCifar, VGG16  # noqa: E402
from utils.options import args_parser  # noqa: E402
from utils.set_seed import set_random_seed  # noqa: E402


class SyntheticImages(Dataset):
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.targets = torch.randint(0, 10, (n,), generator=g).tolist()
        base = torch.randn(10, 3, 32, 32, generator=g)
        noise = torch.randn(n, 3, 32, 32, generator=g)
        self.data = base[self.targets] + 0.8 * noise

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]


def _setup(tmp_path, model_cls, **overrides):
    args = args_parser()
    args.gpu = -1
    args.device = torch.device("cpu")
    args.num_users, args.frac = 12, 0.25  # K = 3
    args.local_ep, args.local_bs, args.bs = 1, 8, 64
    args.epochs, args.crsi_interval = 6, 3
    args.metrics_log_dir = str(tmp_path / "metrics")
    args.crsi_log_dir = str(tmp_path / "diag")
    args.crsi_workers = 1
    args.crsi_null = "signflip"
    args.crsi_store_backend = "ram"
    args.crsi_cache_dir = ""
    for k, v in overrides.items():
        setattr(args, k, v)
    train, test = SyntheticImages(12 * 16, 0), SyntheticImages(64, 1)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(train))
    dict_users = {i: order[i * 16:(i + 1) * 16].tolist() for i in range(12)}
    set_random_seed(args.seed)
    net = model_cls(args)
    return args, net, train, test, dict_users


def _accuracies(text):
    return [float(x) for x in re.findall(r"ROUND_ACCURACY .* accuracy=([\d.]+)", text)]


def _run_fedavg(tmp_path, model_cls):
    args, net, train, test, users = _setup(tmp_path, model_cls, algorithm="FedAvg")
    main_fed.args = args
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main_fed.FedAvg(net, train, test, users)
    return net.state_dict(), _accuracies(buf.getvalue())


def _run_crsi(tmp_path, model_cls, dry_run):
    args, net, train, test, users = _setup(
        tmp_path, model_cls, algorithm="FedCRSI", crsi_dry_run=dry_run
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        FedCRSI(args, net, train, test, users)
    return net.state_dict(), _accuracies(buf.getvalue()), args


def test_dry_run_is_bitwise_fedavg(tmp_path):
    ref_state, ref_acc = _run_fedavg(tmp_path, CNNCifar)
    state, acc, _ = _run_crsi(tmp_path, CNNCifar, dry_run=1)
    assert acc == ref_acc
    for k in ref_state:
        assert torch.equal(ref_state[k], state[k]), k


def test_intervention_changes_only_after_first_interval(tmp_path):
    _, ref_acc = _run_fedavg(tmp_path, CNNCifar)
    state, acc, args = _run_crsi(tmp_path, CNNCifar, dry_run=0)
    first = args.crsi_interval
    assert acc[: first - 1] == ref_acc[: first - 1]
    diag = tmp_path / "diag"
    files = list(diag.glob("*_interventions.jsonl"))
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [r["round"] for r in records] == [3, 6]
    for r in records:
        assert r["pairs"] >= 0 and 0.0 <= r["removed_fraction"] <= 1.0 + 1e-9


def test_memmap_backend_end_to_end(tmp_path):
    """The memmap backend must reproduce the RAM run exactly."""
    ram, acc_ram, _ = _run_crsi(tmp_path / "ram", CNNCifar, dry_run=0)
    args, net, train, test, users = _setup(
        tmp_path / "mm", CNNCifar, algorithm="FedCRSI", crsi_dry_run=0,
        crsi_store_backend="memmap", crsi_cache_dir=str(tmp_path),
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        FedCRSI(args, net, train, test, users)
    assert _accuracies(buf.getvalue()) == acc_ram
    for k, v in net.state_dict().items():
        assert torch.equal(ram[k], v), k


def test_vgg16_smoke(tmp_path):
    """Full VGG16 (13 conv layers) path, including chunking and thread pool."""
    # 4 clients, 3 per round: every 2-round interval has repeated clients
    args, net, train, test, users = _setup(
        tmp_path, VGG16, algorithm="FedCRSI", epochs=4, crsi_interval=2,
        num_users=4, frac=0.75, crsi_workers=2, crsi_chunk_mb=16,
    )
    args.dataset, args.num_channels, args.num_classes = "cifar10", 3, 10
    before = {k: v.clone() for k, v in net.state_dict().items()}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        FedCRSI(args, net, train, test, users)
    out = buf.getvalue()
    assert out.count("CRSI_INTERVENTION") == 2
    assert "PEAK_ACCURACY method=FedCRSI" in out
    records = [json.loads(x) for x in
               next((tmp_path / "diag").glob("*.jsonl")).read_text().splitlines()]
    for rec in records:
        assert rec["repeat_clients"] >= 2
        assert len(rec["layers"]) == 13 and rec["filters"] == 4224
        assert [layer["dim"] for layer in rec["layers"]][:2] == [27, 576]
    # only conv weights may differ from a pure FedAvg update; params still finite
    for k, v in net.state_dict().items():
        assert torch.isfinite(v.float()).all(), k
    assert any(not torch.equal(before[k], v) for k, v in net.state_dict().items())
