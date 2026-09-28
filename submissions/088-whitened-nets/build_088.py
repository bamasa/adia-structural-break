"""resources088: #39 (resources084) plus the whitened member v3 — two rankers, a classifier with context, a pool of whitened trajectory nets."""
import os, joblib, numpy as np
m = joblib.load("resources084/model.joblib")
assert "union_classifier" in m and len(m["nets"]) == 24 and "white3_classifier" not in m
m["white2_ranker"] = joblib.load("resources087_white_rank.joblib")
m["white3_ranker_ctx"] = joblib.load("resources088_white_rank_ctx.joblib")
m["white3_classifier"] = joblib.load("resources088_white_clf.joblib")
arc = np.load("resources088_white_nets.npz"); n = 1 + max(int(k.split("|")[0]) for k in arc.files)
m["white_nets"] = [{k.split("|", 1)[1]: arc[k] for k in arc.files if int(k.split("|")[0]) == i} for i in range(n)]
m["white_net_mu"] = np.load("mu_w.npy"); m["white_net_sd"] = np.load("sd_w.npy")
assert all(s["inp.weight"].shape[1] == 111 for s in m["white_nets"]) and len(m["white_net_mu"]) == 111
os.makedirs("resources088", exist_ok=True)
joblib.dump(m, "resources088/model.joblib", compress=3)
print(f"resources088: nets {len(m['nets'])}, white nets {len(m['white_nets'])}, clf {m['booster'].booster_.num_feature()}, white clf {m['white3_classifier'].booster_.num_feature()}, "
      f"white ranks {m['white2_ranker'].booster_.num_feature()}/{m['white3_ranker_ctx'].booster_.num_feature()}, {os.path.getsize('resources088/model.joblib')/1e6:.1f} MB")
