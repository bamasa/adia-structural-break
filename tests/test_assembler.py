"""The assembler must refuse a submission whose channel widths disagree.

Submission #25 died in the cloud on a one-line mismatch between the interface
and the artifact; the check that now guards it is exercised here on fake
artifacts, group by group.  Run with:  python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from assemble_submission import verify_channels  # noqa: E402


class _Booster:
    def __init__(self, n):
        self._n = n

    def num_feature(self):
        return self._n


def _artifact(clf, ranks, nets, mu):
    return {"booster": _Booster(clf), "rankers": [_Booster(n) for n in ranks],
            "nets": [{"inp.weight": np.zeros((64, n, 1))} for n in nets],
            "net_mu": np.zeros(mu), "net_sd": np.ones(mu)}


class ChannelCheck(unittest.TestCase):
    def _run(self, text, artifact):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "main.py"
            if artifact is not None:
                os.makedirs(Path(d) / "resources")
                joblib.dump(artifact, Path(d) / "resources" / "model.joblib")
            verify_channels(text, target)

    def test_uniform_widths_pass(self):
        self._run("NET_CHANNELS = 200\n", _artifact(200, [200, 200], [200] * 3, 200))

    def test_per_group_widths_pass(self):
        text = "NET_CHANNELS = 200\nRANK_CHANNELS = 200\nCLF_CHANNELS = 206\n"
        self._run(text, _artifact(206, [200], [200] * 12, 200))

    def test_the_25_mismatch_is_refused(self):
        with self.assertRaises(SystemExit):
            self._run("NET_CHANNELS = 186\n", _artifact(200, [200], [200] * 8, 200))

    def test_classifier_width_is_checked_separately(self):
        text = "NET_CHANNELS = 200\nCLF_CHANNELS = 206\n"
        with self.assertRaises(SystemExit):
            self._run(text, _artifact(200, [200], [200], 200))   # classifier built at 200, declared 206

    def test_normalisation_length_is_checked(self):
        with self.assertRaises(SystemExit):
            self._run("NET_CHANNELS = 200\n", _artifact(200, [200], [200], 186))

    def test_classifier_list_is_checked(self):
        text = "NET_CHANNELS = 200\nCLF_CHANNELS = 206\n"
        art = _artifact(206, [200], [200] * 2, 200)
        art["classifiers"] = [_Booster(206), _Booster(206)]
        self._run(text, art)
        art["classifiers"] = [_Booster(206), _Booster(200)]   # a wide model built at the wrong width
        with self.assertRaises(SystemExit):
            self._run(text, art)

    def test_member_suffix_widths_are_checked(self):
        text = "NET_CHANNELS = 200\nCLF_CHANNELS = 206\nMASS_OFFSET = 206\nFREQDEP_OFFSET = 296\nNOVELTY_OFFSET = 396\n"
        art = _artifact(206, [200], [200], 200)
        art.update(mass_classifier=_Booster(90), freqdep_classifier=_Booster(100), union_classifier=_Booster(120))
        self._run(text, art)
        art["white3_classifier"], art["white2_ranker"], art["white3_ranker_ctx"] = _Booster(119), _Booster(111), _Booster(119)
        art["white_nets"] = [{"inp.weight": np.zeros((64, 111, 1))}]; art["white_net_mu"] = np.zeros(111); art["white_net_sd"] = np.ones(111)
        decl = text + "WHITE_OFFSET = 416\nWHITE_RANK_WIDTH = 111\nWHITE_CLF_WIDTH = 119\nWHITE_NET_WIDTH = 111\n"
        self._run(decl, art)
        art["white_nets"] = [{"inp.weight": np.zeros((64, 119, 1))}]   # a net built on the context too
        with self.assertRaises(SystemExit):
            self._run(decl, art)
        for key in ("white3_classifier", "white2_ranker", "white3_ranker_ctx", "white_nets", "white_net_mu", "white_net_sd"):
            del art[key]
        art["white2_classifier"], art["white2_ranker"] = _Booster(111), _Booster(111)
        self._run(text + "WHITE_OFFSET = 416\nWHITE2_WIDTH = 111\n", art)
        art["white2_ranker"] = _Booster(90)       # a ranker built without the odds
        with self.assertRaises(SystemExit):
            self._run(text + "WHITE_OFFSET = 416\nWHITE2_WIDTH = 111\n", art)
        del art["white2_classifier"], art["white2_ranker"]
        art["white_classifier"] = _Booster(90)
        self._run(text + "WHITE_OFFSET = 416\n", art)
        art["white_classifier"] = _Booster(76)    # the full battery without the extras
        with self.assertRaises(SystemExit):
            self._run(text + "WHITE_OFFSET = 416\n", art)
        del art["white_classifier"]
        art["dep_classifier"] = _Booster(23)
        self._run(text + "DEP_OFFSET = 416\n", art)
        art["dep_classifier"] = _Booster(20)      # the novelty twenty by mistake
        with self.assertRaises(SystemExit):
            self._run(text + "DEP_OFFSET = 416\n", art)
        del art["dep_classifier"]
        art["union_classifier"] = _Booster(100)   # built on the frequency hundred alone
        with self.assertRaises(SystemExit):
            self._run(text, art)

    def test_no_artifact_is_not_an_error(self):
        self._run("NET_CHANNELS = 200\n", None)


if __name__ == "__main__":
    unittest.main()
