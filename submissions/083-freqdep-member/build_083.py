"""resources083: #37 (resources082) plus the frequency/dependence classifier."""
import os, joblib
m = joblib.load("resources082/model.joblib")
assert "mass_classifier" in m and len(m["nets"]) == 24 and "freqdep_classifier" not in m
m["freqdep_classifier"] = joblib.load("resources083_freqdep.joblib")
os.makedirs("resources083", exist_ok=True)
joblib.dump(m, "resources083/model.joblib", compress=3)
print(f"resources083: nets {len(m['nets'])}, clf {m['booster'].booster_.num_feature()}, mass {m['mass_classifier'].booster_.num_feature()}, "
      f"freqdep {m['freqdep_classifier'].booster_.num_feature()}, {os.path.getsize('resources083/model.joblib')/1e6:.1f} MB")
