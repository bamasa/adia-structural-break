"""resources084: #38 (resources083) plus the frequency+novelty union classifier."""
import os, joblib
m = joblib.load("resources083/model.joblib")
assert "freqdep_classifier" in m and len(m["nets"]) == 24 and "union_classifier" not in m
m["union_classifier"] = joblib.load("resources084_union.joblib")
os.makedirs("resources084", exist_ok=True)
joblib.dump(m, "resources084/model.joblib", compress=3)
print(f"resources084: nets {len(m['nets'])}, clf {m['booster'].booster_.num_feature()}, mass {m['mass_classifier'].booster_.num_feature()}, "
      f"freqdep {m['freqdep_classifier'].booster_.num_feature()}, union {m['union_classifier'].booster_.num_feature()}, {os.path.getsize('resources084/model.joblib')/1e6:.1f} MB")
