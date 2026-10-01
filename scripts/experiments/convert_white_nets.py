"""Convert the whitened-channel nets .pt -> npz (separate process: torch and lightgbm crash when loaded together)."""
import sys, glob, numpy as np, torch
members = sorted(glob.glob("nets_white_w/member_w[0-9].pt"))
arc = {}
for i, path in enumerate(members):
    sd = torch.load(path, map_location="cpu")
    for k, v in sd.items(): arc[f"{i}|{k}"] = v.numpy().astype("float32")
np.savez("resources088_white_nets.npz", **arc); print(f"nets {len(members)}: {[p.split('/')[-1] for p in members]}, input {arc['0|inp.weight'].shape[1]}")
