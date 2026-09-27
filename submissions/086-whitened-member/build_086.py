"""resources086: #39 (resources084) plus the whitened-stream classifier."""
import os, joblib
m = joblib.load("resources084/model.joblib")
assert "union_classifier" in m and len(m["nets"]) == 24 and "white_classifier" not in m
m["white_classifier"] = joblib.load("resources086_white.joblib")
os.makedirs("resources086", exist_ok=True)
joblib.dump(m, "resources086/model.joblib", compress=3)
print(f"resources086: nets {len(m['nets'])}, clf {m['booster'].booster_.num_feature()}, mass {m['mass_classifier'].booster_.num_feature()}, "
      f"freqdep {m['freqdep_classifier'].booster_.num_feature()}, union {m['union_classifier'].booster_.num_feature()}, white {m['white_classifier'].booster_.num_feature()}, {os.path.getsize('resources086/model.joblib')/1e6:.1f} MB")
