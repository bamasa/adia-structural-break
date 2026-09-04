"""torch-only step: aug3 checkpoints -> one numpy archive (torch + lightgbm in one process segfaults on macOS)."""
import glob, os, numpy as np, torch
paths = sorted(glob.glob("nets_aug3_full/member_f*.pt"))
out = {}
for i, p in enumerate(paths):
    for k, v in torch.load(p, map_location="cpu").items():
        out[f"{i}|{k}"] = v.numpy()
np.savez("resources074/nets_aug3.npz", **out)
print(f"{len(paths)} nets converted: {', '.join(os.path.basename(p) for p in paths)}")
