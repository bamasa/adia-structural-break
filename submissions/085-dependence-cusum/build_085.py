"""resources085: #39 (resources084) plus the dependence-CUSUM classifier."""
import os, joblib
m = joblib.load("resources084/model.joblib")
assert "union_classifier" in m and len(m["nets"]) == 24 and "dep_classifier" not in m
m["dep_classifier"] = joblib.load("resources085_dep.joblib")
os.makedirs("resources085", exist_ok=True)
joblib.dump(m, "resources085/model.joblib", compress=3)
print(f"resources085: nets {len(m['nets'])}, clf {m['booster'].booster_.num_feature()}, mass {m['mass_classifier'].booster_.num_feature()}, "
      f"freqdep {m['freqdep_classifier'].booster_.num_feature()}, union {m['union_classifier'].booster_.num_feature()}, dep {m['dep_classifier'].booster_.num_feature()}, {os.path.getsize('resources085/model.joblib')/1e6:.1f} MB")
