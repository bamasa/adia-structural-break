"""Streaming structural-break detection for univariate time series.

The decisive component is the whitened stream (:mod:`structural_break.white`):
``WhiteMonitor(history)`` fits an AR(p) by BIC, a conditional scale and the
empirical CDF of the innovations on the break-free history, and then turns
every new online point into statistics that are i.i.d. N(0,1) under the null
-- CUSUMs, generalised likelihood ratios over dyadic windows, one-sample
tests, and (with ``odds=True``) Shiryaev-Roberts odds of a change at some
earlier step. The other modules are the channel families of the competition
ensemble; ``combiners.ts_auc`` is the competition metric.

    >>> import numpy as np
    >>> from structural_break import WhiteMonitor, channel_names
    >>> wm = WhiteMonitor(np.random.default_rng(0).standard_normal(2000), odds=True)
    >>> names = channel_names(odds=True)
    >>> row = wm.update(0.3)            # one online point -> one channel vector
    >>> len(row) == len(names)
    True
"""

from .white import (
    CONTEXT_CHANNELS,
    SR_CHANNELS,
    SR_MEANS,
    SR_PHIS,
    SR_VAR_DOWN,
    SR_VAR_UP,
    WHITE_CHANNELS,
    WHITE_DYADIC,
    WHITE_LAGS,
    WHITE_ODDS_CHANNELS,
    WHITE_TESTW,
    WhiteMonitor,
)

__all__ = ["WhiteMonitor", "channel_names", "WHITE_CHANNELS", "WHITE_ODDS_CHANNELS", "SR_CHANNELS", "CONTEXT_CHANNELS", "__version__"]
__version__ = "0.1.0"


def channel_names(odds: bool = False, context: bool = False) -> list[str]:
    """Names of the channels ``WhiteMonitor.update`` returns, in order."""
    names: list[str] = []
    for stream in ("u",):
        names += [f"{stream}_mean_cusum", f"{stream}_mean_ewma02", f"{stream}_mean_ewma005", f"{stream}_mean_prefix_z",
                  f"{stream}_scale_cusum", f"{stream}_scale_ewma02", f"{stream}_scale_ewma005", f"{stream}_scale_prefix_z"]
        for k in WHITE_LAGS:
            names += [f"{stream}_dep{k}_cusum", f"{stream}_dep{k}_ewma01"]
        names += [f"{stream}_portmanteau", f"{stream}_vol_cusum", f"{stream}_vol_ewma01",
                  f"{stream}_skew_ewma", f"{stream}_kurt_ewma", f"{stream}_tail25_ewma", f"{stream}_tail15_ewma",
                  f"{stream}_core03_ewma", f"{stream}_signchange_ewma", f"{stream}_absmean_ewma"]
        for w in WHITE_DYADIC:
            names += [f"{stream}_glr_mean_w{w}", f"{stream}_glr_scale_w{w}", f"{stream}_glr_dep_w{w}"]
        names += [f"{stream}_glr_mean_max", f"{stream}_glr_scale_max", f"{stream}_glr_dep_max"]
        for w in list(WHITE_TESTW) + ["prefix"]:
            names += [f"{stream}_ks_w{w}", f"{stream}_cvm_w{w}", f"{stream}_ad_w{w}"]
        names += [f"{stream}_ks_max", f"{stream}_cvm_max", f"{stream}_ad_max"]
    names += ["c_mean_cusum", "c_mean_ewma02", "c_dep1_cusum", "c_dep2_cusum", "c_dep5_cusum", "c_portmanteau",
              "c_glr_mean_max", "c_glr_dep_max", "c_ks_max", "c_ad_max", "logscale", "logscale_ewma02", "logscale_cusum", "step"]
    assert len(names) == WHITE_CHANNELS, (len(names), WHITE_CHANNELS)
    if odds:
        names += [f"sr_var_up_{v}" for v in SR_VAR_UP] + [f"sr_var_down_{v}" for v in SR_VAR_DOWN]
        for d in SR_MEANS:
            names += [f"sr_mean_+{d}", f"sr_mean_-{d}"]
        for phi in SR_PHIS:
            names += [f"sr_dep_+{phi}", f"sr_dep_-{phi}"]
        names += ["sr_mix_var_up", "sr_mix_var_down", "sr_mix_mean", "sr_mix_dep"]
        assert len(names) == WHITE_ODDS_CHANNELS, (len(names), WHITE_ODDS_CHANNELS)
    if context:
        names += ["ctx_ar_order", "ctx_ar_coef1", "ctx_scale_lambda", "ctx_log_innov_var", "ctx_innov_kurt", "ctx_innov_skew", "ctx_log_hist_len", "ctx_max_abs_z"]
    return names
