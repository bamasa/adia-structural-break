"""The competition metric against sklearn's AUC, step by step.

``ts_auc`` is the time-stratified AUC the platform scores with: at each step an
ordinary AUC over the series alive there, averaged with weights n_pos * n_neg,
steps with a single class skipped. The platform's per-step AUC is
``sklearn.metrics.roc_auc_score``, which midranks ties, so the reference here is
built from it directly; the cases without sklearn pin the tie, single-class and
empty-weight behaviour by hand. Run with:
PYTHONPATH=src python -m unittest discover -s tests
"""

import unittest

import numpy as np

from structural_break.combiners import ts_auc

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover - the optional check is skipped
    roc_auc_score = None


def _reference(scores, labels, steps):
    """Per-step roc_auc_score, weighted by the number of positive-negative pairs."""
    weighted = total = 0.0
    for step in np.unique(steps):
        mask = steps == step
        y = labels[mask]
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            continue
        weighted += n_pos * n_neg * roc_auc_score(y, scores[mask])
        total += n_pos * n_neg
    return weighted / total if total else 0.5


def _panel(rng, n_series=40, max_steps=30, levels=None):
    """Scores, labels and steps of a panel whose series end at different steps.

    One label per series, as in the competition; the series lengths differ, so the
    late steps hold few series and some of them a single class. ``levels``
    quantises the scores to that many values, which makes ties common.
    """
    scores, labels, steps = [], [], []
    for _ in range(n_series):
        length = int(rng.integers(5, max_steps + 1))
        s = rng.random(length)
        if levels:
            s = np.round(s * levels) / levels
        scores.append(s)
        labels.append(np.full(length, int(rng.random() < 0.5)))
        steps.append(np.arange(length))
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(steps)


def _both_classes(labels, steps):
    """Mask of the rows whose step has both classes."""
    keep = np.zeros(len(steps), dtype=bool)
    for step in np.unique(steps):
        mask = steps == step
        if 0 < labels[mask].sum() < mask.sum():
            keep |= mask
    return keep


@unittest.skipUnless(roc_auc_score is not None, "scikit-learn is not installed")
class AgainstSklearn(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_random_scores(self):
        scores, labels, steps = _panel(self.rng)
        self.assertAlmostEqual(ts_auc(scores, labels, steps), _reference(scores, labels, steps), places=12)

    def test_many_ties(self):
        scores, labels, steps = _panel(self.rng, n_series=60, levels=4)
        self.assertLessEqual(len(np.unique(scores)), 5)      # the ties are real
        self.assertAlmostEqual(ts_auc(scores, labels, steps), _reference(scores, labels, steps), places=12)

    def test_steps_with_a_single_class_are_skipped(self):
        scores, labels, steps = _panel(self.rng, n_series=12, max_steps=40)
        keep = _both_classes(labels, steps)
        self.assertTrue((~keep).any())                       # some late steps hold one class
        self.assertAlmostEqual(ts_auc(scores, labels, steps), _reference(scores, labels, steps), places=12)
        self.assertAlmostEqual(ts_auc(scores, labels, steps), ts_auc(scores[keep], labels[keep], steps[keep]), places=12)

    def test_step_with_all_equal_scores_contributes_half(self):
        scores, labels, steps = _panel(self.rng)
        flat_scores = np.full(6, 0.5)
        flat_labels = np.array([1, 1, 1, 0, 0, 0])
        flat_steps = np.full(6, 1000)
        self.assertEqual(roc_auc_score(flat_labels, flat_scores), 0.5)
        scores = np.concatenate([scores, flat_scores])
        labels = np.concatenate([labels, flat_labels])
        steps = np.concatenate([steps, flat_steps])
        self.assertAlmostEqual(ts_auc(scores, labels, steps), _reference(scores, labels, steps), places=12)


class ByHand(unittest.TestCase):
    def test_ties_are_midranked(self):
        # The tied positive shares rank 2.5 with a negative: half a pair, AUC 0.75.
        scores = np.array([1.0, 1.0, 0.0])
        labels = np.array([1, 0, 0])
        steps = np.zeros(3, dtype=int)
        self.assertAlmostEqual(ts_auc(scores, labels, steps), 0.75, places=12)

    def test_all_equal_scores_is_half(self):
        scores = np.full(5, 0.3)
        labels = np.array([1, 0, 1, 0, 0])
        steps = np.zeros(5, dtype=int)
        self.assertEqual(ts_auc(scores, labels, steps), 0.5)

    def test_no_step_with_both_classes_returns_half(self):
        scores = np.array([0.1, 0.9, 0.4, 0.6])
        labels = np.array([1, 1, 0, 0])
        steps = np.array([0, 0, 1, 1])
        self.assertEqual(ts_auc(scores, labels, steps), 0.5)

    def test_steps_are_weighted_by_pair_counts(self):
        # Step 0: one pair, AUC 1. Step 1: four pairs, all scores equal, AUC 0.5.
        scores = np.array([0.9, 0.1, 0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 0, 1, 1, 0, 0])
        steps = np.array([0, 0, 1, 1, 1, 1])
        self.assertAlmostEqual(ts_auc(scores, labels, steps), (1 * 1.0 + 4 * 0.5) / 5, places=12)


if __name__ == "__main__":
    unittest.main()
