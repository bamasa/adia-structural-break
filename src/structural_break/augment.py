"""Manufacturing training windows by cutting the labelled series.

The training set holds ten thousand series but only five thousand breaks, and a
window classifier is hungrier than that. The series themselves contain far more
supervision than one label each: every break-free stretch is a factory of
negatives, and every true break can sit at any offset inside a window — each
offset a distinct example of what "a break somewhere in view" looks like.

What gets cut
-------------
**Negatives** come from stretches guaranteed break-free: anywhere in the
history, and the online segment strictly before tau. A window is drawn at a
random position and a random *stride* — taking every s-th observation of a
stretch s times the window long. The stride is the "thinning" in the design:
it varies the effective time-scale of a window without changing its length, so
the classifier meets slow drift and fast jitter as the same shaped input.

**Positives** straddle a true tau, with the break placed at a random offset
between 15% and 85% of the window — never at a fixed position, or the
classifier learns the position instead of the break. The same stride
augmentation applies (the "resampling"), which multiplies each of the five
thousand real breaks into dozens of distinct training views.

What is deliberately absent
---------------------------
Synthetic breaks. Splicing unrelated segments together would manufacture
unlimited positives, but their break types would be *ours* rather than the
organisers' — and a classifier trained on home-made breaks learns the seams of
the splicing, not the structure of the task. Every positive here contains a
break the organisers wrote.

Windows are standardised by the history of their own series before cutting, so
the classifier never sees raw scale, and clipped at the series' winsor level —
the same view the deployed monitor produces at inference time. An augmentation
that differs from deployment is a validation defect waiting to be scored.
"""

from __future__ import annotations

import numpy as np

from structural_break.cnn import WINDOW
from structural_break.features import Normalisation

#: Strides used for thinning/resampling. 1 is the native scale; 2 and 4 widen
#: the effective receptive field of the same 128-observation window.
STRIDES = (1, 2, 4)


def standardise_series(
    history: np.ndarray, online: np.ndarray
) -> tuple[np.ndarray, int]:
    """The full standardised series and the index where the online part starts."""
    norm = Normalisation.fit(history)
    h = np.asarray(
        [
            norm.clip(norm.standardise(float(v), -len(history) + i))
            for i, v in enumerate(history)
        ]
    )
    o = np.asarray(
        [norm.clip(norm.standardise(float(v), i)) for i, v in enumerate(online)]
    )
    return np.concatenate([h, o]), len(h)


def cut(values: np.ndarray, end: int, stride: int) -> np.ndarray | None:
    """A window of ``WINDOW`` points ending at ``end`` (exclusive), thinned."""
    span = WINDOW * stride
    start = end - span
    if start < 0:
        return None
    return values[start:end:stride].astype("float32")


def windows_for_series(
    full: np.ndarray,
    online_start: int,
    tau: int | None,
    rng: np.random.Generator,
    *,
    per_kind: int = 8,
) -> tuple[list[np.ndarray], list[int]]:
    """Cut one series into labelled windows.

    ``tau`` is the break position within the online segment, or None. Negatives
    are cut so the whole *span* (window times stride) sits inside break-free
    data; positives so the break falls at a controlled offset inside the view.
    """
    windows: list[np.ndarray] = []
    labels: list[int] = []
    break_at = online_start + tau if tau is not None else None
    total = len(full)

    for stride in STRIDES:
        span = WINDOW * stride
        # --- negatives: any end position whose span avoids the break ---------
        safe_end = break_at if break_at is not None else total
        if safe_end >= span:
            for _ in range(per_kind):
                end = int(rng.integers(span, safe_end + 1))
                w = cut(full, end, stride)
                if w is not None:
                    windows.append(w)
                    labels.append(0)
        # --- positives: the break at 15-85% of the window --------------------
        if break_at is not None:
            for _ in range(per_kind):
                offset = int(rng.integers(int(0.15 * WINDOW), int(0.85 * WINDOW)))
                end = break_at + (WINDOW - offset) * stride
                if end > total:
                    continue
                w = cut(full, end, stride)
                if w is not None:
                    windows.append(w)
                    labels.append(1)
    return windows, labels
