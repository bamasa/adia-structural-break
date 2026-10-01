"""resources080: the #30 artifact (resources075) with the net pool extended from 12 to 24.

The same twelve additional members as in #34 (six on all series from #29, six from the 095 pool),
but without self-normalization and without the second classifier — neither was confirmed in the cloud.
The only difference from #30 is the number of nets.
"""
import os, joblib, numpy as np
m = joblib.load("resources075/model.joblib")
assert "classifiers" not in m and len(m["nets"]) == 12, (list(m), len(m["nets"]))
def unpack(path):
    arc = np.load(path); n = 1 + max(int(k.split("|")[0]) for k in arc.files)
    return [{k.split("|", 1)[1]: arc[k] for k in arc.files if int(k.split("|")[0]) == i} for i in range(n)]
extra = unpack("resources074/nets_aug3.npz") + unpack("resources079/nets_pool6.npz")
assert len(extra) == 12 and all(s["inp.weight"].shape[1] == 200 for s in extra)
m["nets"] = list(m["nets"]) + extra
os.makedirs("resources080", exist_ok=True)
joblib.dump(m, "resources080/model.joblib", compress=3)
print(f"resources080: nets {len(m['nets'])}, rankers {len(m['rankers'])}, clf {m['booster'].booster_.num_feature()}, "
      f"{os.path.getsize('resources080/model.joblib')/1e6:.1f} MB")
