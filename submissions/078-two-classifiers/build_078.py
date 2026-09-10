"""resources078: resources075 + the wide classifier."""
import os, joblib
m = joblib.load("resources075/model.joblib")
wide = joblib.load("resources077_wide206.joblib")
m["classifiers"] = [m["booster"], wide]
os.makedirs("resources078", exist_ok=True)
joblib.dump(m, "resources078/model.joblib", compress=3)
print(f"resources078: classifiers {[c.booster_.num_feature() for c in m['classifiers']]}, nets {len(m['nets'])}, {os.path.getsize('resources078/model.joblib')/1e6:.1f} MB")
