"""Preflight for the GPU job: verify the box, the matrices, and the speed.

Run this first. It answers three questions in under a minute — is CUDA
really there, are all eleven matrices present and mutually consistent, and
how long one member will take — so a mistake costs seconds instead of an
hour of silent training.

    python scripts/gpu/check_environment.py --data-dir /data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

MATRICES = {
    "X40.npy": 40, "C_cnn.npy": 1, "N9.npy": 9, "X50a.npy": 50,
    "E4.npy": 4, "B40.npy": 42, "B2.npy": 40, "SPEC14.npy": 14,
}
LABELS = ("Y40.npy", "G40.npy", "S40.npy")
#: Triple boundary augmentation — its own row count, checked for shape only.
AUGMENTATION = {"AUG3_X.npy": 200, "AUG3_Y.npy": None, "AUG3_G.npy": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    args = ap.parse_args()
    ok = True

    try:
        import torch
    except ImportError:
        print("FAIL  torch is not installed — see scripts/gpu/README.md")
        return 1
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"OK    CUDA: {name}, {mem:.0f} GB")
    else:
        print("WARN  no CUDA — the job will run, slowly, on the CPU")

    total_cols = 0
    n_rows = None
    for name, cols in MATRICES.items():
        path = args.data_dir / name
        if not path.exists():
            print(f"FAIL  missing {name}")
            ok = False
            continue
        arr = np.load(path, mmap_mode="r")
        rows = arr.shape[0]
        got = arr.shape[1] if arr.ndim > 1 else 1
        if n_rows is None:
            n_rows = rows
        if rows != n_rows:
            print(f"FAIL  {name}: {rows} rows, expected {n_rows}")
            ok = False
        elif got != cols:
            print(f"FAIL  {name}: {got} columns, expected {cols}")
            ok = False
        else:
            total_cols += cols
    for name in LABELS:
        if not (args.data_dir / name).exists():
            print(f"FAIL  missing {name}")
            ok = False

    aug_rows = None
    for name, cols in AUGMENTATION.items():
        path = args.data_dir / name
        if not path.exists():
            print(f"FAIL  missing {name} — the nets are data-bound; this file is the lever")
            ok = False
            continue
        arr = np.load(path, mmap_mode="r")
        if aug_rows is None:
            aug_rows = arr.shape[0]
        if arr.shape[0] != aug_rows:
            print(f"FAIL  {name}: {arr.shape[0]} rows, expected {aug_rows}")
            ok = False
        elif cols is not None and arr.shape[1] != cols:
            print(f"FAIL  {name}: {arr.shape[1]} columns, expected {cols}")
            ok = False

    if not ok:
        print("\nSomething is missing — copy the files listed in scripts/gpu/README.md")
        return 1
    print(f"OK    matrices: {n_rows:,} rows x {total_cols} channels, labels present")
    print(f"OK    augmentation: {aug_rows:,} rows of triple boundary augmentation")

    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = nn.Sequential(nn.Conv1d(200, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1)).to(device)
    x = torch.randn(48, 200, 512, device=device)
    opt = torch.optim.Adam(net.parameters())
    for _ in range(3):                       # warm-up
        opt.zero_grad(); net(x).mean().backward(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        opt.zero_grad(); net(x).mean().backward(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_batch = (time.time() - t0) / 20
    # ~200 batches per epoch of originals; the augmentation multiplies the rows.
    batches_per_epoch = 200 * (n_rows + aug_rows) / n_rows
    est_member = per_batch * batches_per_epoch * 10 / 60
    print(f"OK    speed: {per_batch*1000:.0f} ms per batch, roughly "
          f"{est_member:.0f} min per member")
    print("\nReady. Next:")
    print(f"  python scripts/gpu/train_nets_cuda.py --data-dir {args.data_dir} "
          f"--out-dir nets_cuda --members 24")
    return 0


if __name__ == "__main__":
    sys.exit(main())
