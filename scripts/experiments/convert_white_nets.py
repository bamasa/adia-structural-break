"""Конвертация сетей на отбелённых каналах .pt -> npz (отдельный процесс: torch и lightgbm вместе падают)."""
import sys, glob, numpy as np, torch
members = sorted(glob.glob("nets_white_w/member_w[0-9].pt"))
arc = {}
for i, path in enumerate(members):
    sd = torch.load(path, map_location="cpu")
    for k, v in sd.items(): arc[f"{i}|{k}"] = v.numpy().astype("float32")
np.savez("resources088_white_nets.npz", **arc); print(f"сетей {len(members)}: {[p.split('/')[-1] for p in members]}, вход {arc['0|inp.weight'].shape[1]}")
