"""Build a self-contained main.py for the platform from the library sources.

The platform wants one file; the repository wants one source of truth. Copying
code into submissions by hand is how the two drift apart — the drift is
invisible until a cloud run disagrees with a local one, and the disagreement
costs a submission slot to discover. So the submission is *assembled*: library
modules concatenated in dependency order, their internal imports and module
docstrings stripped, the competition interface appended. The library stays the
only place code is edited.
"""

from __future__ import annotations

import re
import sys

#: Channels the mass battery emits (src/structural_break/mass.py).
MASS_CHANNELS_CONST = 90
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "structural_break"

#: Dependency order matters: later modules use earlier names.
MODULES = ["features.py", "detectors.py", "retrospective.py", "retro2.py", "multiscale.py", "bocpd.py", "mass.py", "freqdep.py", "novelty.py", "depcusum.py", "white.py"]

HEADER = '''"""{title}

Assembled from the library by scripts/assemble_submission.py — edits belong in
src/structural_break/, never here.

{description}
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import joblib
import json
import numpy as np
from scipy.special import gammaln, ndtr, ndtri

#: One worker per pair of cores. Left unset, the platform runs a single worker
#: on a sixteen-core machine, and quota is billed in wall-clock hours.
INFER_PARALLELISM = 8

'''


def strip(module: str) -> str:
    text = (SRC / module).read_text()
    text = re.sub(r'^""".*?"""\n', "", text, count=1, flags=re.S)
    lines = []
    in_import = False
    for line in text.split("\n"):
        if in_import:
            # A parenthesised import runs until its closing bracket, and every
            # continuation line is part of it — the first version dropped only
            # the opening line and left an orphaned indented block behind.
            if ")" in line:
                in_import = False
            continue
        if line.startswith("from __future__") or line.startswith("import "):
            continue
        if line.startswith("from ") and " import " in line:
            if "(" in line and ")" not in line:
                in_import = True
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def assemble(title: str, description: str, interface: str) -> str:
    parts = [HEADER.format(title=title, description=description)]
    for module in MODULES:
        parts.append(f"# --- {module} " + "-" * (60 - len(module)) + "\n")
        parts.append(strip(module))
    parts.append(interface)
    return "\n\n".join(parts)


def verify_channels(assembled: str, target: Path) -> None:
    """Refuse to ship a submission whose channel counts disagree.

    Submission #25 died in the cloud because the interface still said
    ``NET_CHANNELS = 186`` while every net in the artifact expected 200 — a
    one-line leftover that no local import could catch, since the shapes only
    meet at inference time. The counts all exist before shipping, so the
    assembler compares them: every ``*_CHANNELS`` constant in the assembled
    text against each other, and against the artifact next to the target —
    tree feature counts, net input widths, normalisation vector lengths.
    """
    m = re.search(r"^NET_CHANNELS = (\d+)", assembled, re.M)
    if not m:
        return
    declared = {"NET_CHANNELS": int(m.group(1))}
    # The classifier and the rankers may read a different width (a suffix the
    # nets never see); each group is checked against its own declaration.
    width = {}
    for group, key in (("clf", "CLF_CHANNELS"), ("rank", "RANK_CHANNELS")):
        mm = re.search(rf"^{key} = (\d+)", assembled, re.M)
        width[group] = int(mm.group(1)) if mm else declared["NET_CHANNELS"]
    # A member reading its own suffix declares where that suffix starts.
    mo = re.search(r"^MASS_OFFSET = (\d+)", assembled, re.M)
    width["mass"] = MASS_CHANNELS_CONST if mo else None

    artifact = target.parent / "resources" / "model.joblib"
    if not artifact.exists():
        print(f"interface declares NET_CHANNELS={m.group(1)}; no artifact to check yet")
        return

    import joblib

    model = joblib.load(artifact)
    found, bad = dict(declared), []
    def check(name, got, want):
        found[name] = got
        if got != want:
            bad.append(f"{name}={got} (expected {want})")
    if model.get("mass_classifier") is not None and width.get("mass"):
        check("mass", getattr(model["mass_classifier"], "booster_", model["mass_classifier"]).num_feature(), width["mass"])
    if model.get("freqdep_classifier") is not None and re.search(r"^FREQDEP_OFFSET = ", assembled, re.M):
        check("freqdep", getattr(model["freqdep_classifier"], "booster_", model["freqdep_classifier"]).num_feature(), 100)
    # The union member reads the frequency/dependence hundred and the novelty
    # twenty together; its classifier must have been built at that width.
    if model.get("union_classifier") is not None and re.search(r"^NOVELTY_OFFSET = ", assembled, re.M):
        check("union", getattr(model["union_classifier"], "booster_", model["union_classifier"]).num_feature(), 120)
    if model.get("dep_classifier") is not None and re.search(r"^DEP_OFFSET = ", assembled, re.M):
        check("dep", getattr(model["dep_classifier"], "booster_", model["dep_classifier"]).num_feature(), 23)
    if model.get("white_classifier") is not None and re.search(r"^WHITE_OFFSET = ", assembled, re.M):
        check("white", getattr(model["white_classifier"], "booster_", model["white_classifier"]).num_feature(), 90)
    if model.get("booster") is not None:
        check("clf", getattr(model["booster"], "booster_", model["booster"]).num_feature(), width["clf"])
    for i, c in enumerate(model.get("classifiers", [])):
        check(f"clf[{i}]", getattr(c, "booster_", c).num_feature(), width["clf"])
    for i, r in enumerate(model.get("rankers", [])):
        check(f"rank[{i}]", getattr(r, "booster_", r).num_feature(), width["rank"])
    for i, state in enumerate(model.get("nets", [])):
        check(f"net[{i}]", state["inp.weight"].shape[1], declared["NET_CHANNELS"])
    for key in ("net_mu", "net_sd"):
        if key in model:
            check(key, len(model[key]), declared["NET_CHANNELS"])
    if bad:
        raise SystemExit("channel counts disagree between interface and artifact: " + "; ".join(bad))
    print(f"channel check passed: {len(found)} counts — nets {declared['NET_CHANNELS']}, "
          f"classifier {width['clf']}, rankers {width['rank']}")


if __name__ == "__main__":
    target = Path(sys.argv[1])
    interface = Path(sys.argv[2]).read_text()
    meta = Path(sys.argv[3]).read_text().split("\n---\n")
    text = assemble(meta[0].strip(), meta[1].strip(), interface)
    verify_channels(text, target)
    target.write_text(text)
    print(f"assembled -> {target}")
