"""Assemble resources073: clean classifier, augmented ranker, aug3 nets."""
import glob, os, sys
import joblib, numpy as np
os.makedirs("resources073", exist_ok=True)
clf = joblib.load("resources053/clf200.joblib")
rank_aug = joblib.load("resources070/rank_aug.joblib")
forecaster = joblib.load("resources072/model.joblib")["forecaster"]
arc = np.load("resources073/nets_aug3.npz")
n = 1 + max(int(k.split("|")[0]) for k in arc.files)
nets = [{k.split("|", 1)[1]: arc[k] for k in arc.files if int(k.split("|")[0]) == i} for i in range(n)]
paths = [f"member {i}" for i in range(n)]
model = {"booster": clf, "rankers": [rank_aug], "forecaster": forecaster,
         "nets": nets, "net_mu": np.load("mu200.npy"), "net_sd": np.load("sd200.npy")}
joblib.dump(model, "resources073/model.joblib", compress=3)
print(f"resources073: {len(nets)} nets ({', '.join(os.path.basename(p) for p in paths)}), "
      f"{os.path.getsize('resources073/model.joblib')/1e6:.1f} MB")
