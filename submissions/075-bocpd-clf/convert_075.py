"""torch-only: 12 nets (six from #28 + six last-epoch) -> npz."""
import numpy as np, torch
paths = [f"nets_aug3/member_p{i}.pt" for i in range(6)] + [f"nets_aug3_last/member_z{i}.pt" for i in range(6)]
out = {}
for i, p in enumerate(paths):
    for k, v in torch.load(p, map_location="cpu").items():
        out[f"{i}|{k}"] = v.numpy()
np.savez("resources075/nets12.npz", **out)
print(f"{len(paths)} nets converted")
