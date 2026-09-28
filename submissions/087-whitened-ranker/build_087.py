"""resources087: #39 (resources084) plus the whitened member v2 — a per-step ranker and a slow classifier on 111 channels."""
import os, joblib
m = joblib.load("resources084/model.joblib")
assert "union_classifier" in m and len(m["nets"]) == 24 and "white2_classifier" not in m
m["white2_classifier"] = joblib.load("resources087_white_clf.joblib")
m["white2_ranker"] = joblib.load("resources087_white_rank.joblib")
os.makedirs("resources087", exist_ok=True)
joblib.dump(m, "resources087/model.joblib", compress=3)
print(f"resources087: nets {len(m['nets'])}, clf {m['booster'].booster_.num_feature()}, mass {m['mass_classifier'].booster_.num_feature()}, "
      f"freqdep {m['freqdep_classifier'].booster_.num_feature()}, union {m['union_classifier'].booster_.num_feature()}, "
      f"white2 clf {m['white2_classifier'].booster_.num_feature()} / rank {m['white2_ranker'].booster_.num_feature()}, {os.path.getsize('resources087/model.joblib')/1e6:.1f} MB")
