"""resources079: resources078 (#33) + 6 сетей на всех данных (#29) + 6 членов пула = 24 сети."""
import os, joblib, numpy as np
m = joblib.load("resources078/model.joblib")
def unpack(path):
    arc = np.load(path); n = 1 + max(int(k.split("|")[0]) for k in arc.files)
    return [{k.split("|", 1)[1]: arc[k] for k in arc.files if int(k.split("|")[0]) == i} for i in range(n)]
full6 = unpack("resources074/nets_aug3.npz"); pool6 = unpack("resources079/nets_pool6.npz")
assert all(s["inp.weight"].shape[1] == 200 for s in full6 + pool6)
m["nets"] = list(m["nets"]) + full6 + pool6
os.makedirs("resources079", exist_ok=True)
joblib.dump(m, "resources079/model.joblib", compress=3)
print(f"resources079: nets {len(m['nets'])}, classifiers {[c.booster_.num_feature() for c in m['classifiers']]}, {os.path.getsize('resources079/model.joblib')/1e6:.1f} MB")
