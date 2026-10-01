"""resources081: the #30 artifact plus the mass-battery classifier as a separate member."""
import os, joblib
m = joblib.load("resources075/model.joblib")
assert len(m["nets"]) == 12 and "mass_classifier" not in m
m["mass_classifier"] = joblib.load("resources081_massclf.joblib")
os.makedirs("resources081", exist_ok=True)
joblib.dump(m, "resources081/model.joblib", compress=3)
print(f"resources081: nets {len(m['nets'])}, clf {m['booster'].booster_.num_feature()}, "
      f"mass {m['mass_classifier'].booster_.num_feature()}, {os.path.getsize('resources081/model.joblib')/1e6:.1f} MB")
