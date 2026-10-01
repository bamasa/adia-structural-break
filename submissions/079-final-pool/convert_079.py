"""torch-only: six pool members (k6-k11) -> npz."""
import glob, numpy as np, torch
paths = sorted(p for p in glob.glob("nets_pool24/member_k*.pt") if "_epoch" not in p)
out = {}
for i, p in enumerate(paths):
    for k, v in torch.load(p, map_location="cpu").items():
        out[f"{i}|{k}"] = v.numpy()
np.savez("resources079/nets_pool6.npz", **out)
print(f"{len(paths)} pool nets converted: {[p.split('/')[-1] for p in paths]}")
