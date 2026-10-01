"""resources082: the mass member (#36) plus the full pool of 24 nets (#35).

Two independently justified changes to #30 in one artifact: the mass battery
as a separate member (114, +0.004 on fold 2) and the net pool doubled from 12 to 24
(tested separately as #35). Fold 2 does not distinguish 12 from 24 nets, so the
combination is expected to gain from the mass member; the pool is a bet on lower variance in the cloud.
"""
import os, joblib, numpy as np
m = joblib.load("resources081/model.joblib")          # #36: 12 nets + the mass member
assert "mass_classifier" in m and len(m["nets"]) == 12
def unpack(path):
    arc = np.load(path); n = 1 + max(int(k.split("|")[0]) for k in arc.files)
    return [{k.split("|", 1)[1]: arc[k] for k in arc.files if int(k.split("|")[0]) == i} for i in range(n)]
extra = unpack("resources074/nets_aug3.npz") + unpack("resources079/nets_pool6.npz")
assert len(extra) == 12 and all(s["inp.weight"].shape[1] == 200 for s in extra)
m["nets"] = list(m["nets"]) + extra
os.makedirs("resources082", exist_ok=True)
joblib.dump(m, "resources082/model.joblib", compress=3)
print(f"resources082: nets {len(m['nets'])}, clf {m['booster'].booster_.num_feature()}, "
      f"mass {m['mass_classifier'].booster_.num_feature()}, {os.path.getsize('resources082/model.joblib')/1e6:.1f} MB")
