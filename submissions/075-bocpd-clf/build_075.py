"""resources075: clf206 (all series, BOCPD suffix), rank_aug, forecaster, twelve nets."""
import os, joblib, numpy as np
clf = joblib.load("resources075/clf206.joblib")
rank_aug = joblib.load("resources070/rank_aug.joblib")
forecaster = joblib.load("resources072/model.joblib")["forecaster"]
arc = np.load("resources075/nets12.npz")
n = 1 + max(int(k.split("|")[0]) for k in arc.files)
nets = [{k.split("|", 1)[1]: arc[k] for k in arc.files if int(k.split("|")[0]) == i} for i in range(n)]
model = {"booster": clf, "rankers": [rank_aug], "forecaster": forecaster,
         "nets": nets, "net_mu": np.load("mu200.npy"), "net_sd": np.load("sd200.npy")}
joblib.dump(model, "resources075/model.joblib", compress=3)
print(f"resources075: clf {clf.booster_.num_feature()} features, {len(nets)} nets, "
      f"{os.path.getsize('resources075/model.joblib')/1e6:.1f} MB")
