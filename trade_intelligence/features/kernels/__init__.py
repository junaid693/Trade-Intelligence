"""Pure numerical technical indicator kernels for Trade Intelligence."""

from typing import Dict
import numpy as np

from trade_intelligence.features.types import NumericalCandleArrays
from trade_intelligence.features.kernels.trend import (
    calc_ema,
    calc_ema_slope,
    calc_ema_spread,
    calc_price_ema_ratio,
    compute_trend_features,
)
from trade_intelligence.features.kernels.momentum import (
    calc_macd,
    calc_roc,
    calc_rsi,
    compute_momentum_features,
)
from trade_intelligence.features.kernels.volatility import (
    calc_atr,
    calc_bollinger_bands,
    calc_natr,
    calc_true_range,
    compute_volatility_features,
)
from trade_intelligence.features.kernels.volume import (
    calc_rvol,
    calc_taker_buy_share,
    calc_volume_sma,
    compute_volume_features,
)


def compute_all_kernels(arrays: NumericalCandleArrays) -> Dict[str, np.ndarray]:
    """Compute all 30 Unit 2.2 technical feature arrays from a contiguous NumericalCandleArrays segment.

    Features computed (exactly 30):
    - Trend (12): ema20, ema50, ema200, ratio_close_ema20, ratio_close_ema50, ratio_close_ema200,
                  spread_ema20_50, spread_ema50_200, spread_ema20_200, slope_ema20, slope_ema50, slope_ema200
    - Momentum (8): rsi14, macd_line, macd_signal, macd_hist, macd_norm, macd_hist_norm, roc10, roc21
    - Volatility (7): atr14, natr14, bb_upper, bb_mid, bb_lower, bb_pct_b, bb_bandwidth
    - Volume (3): vol_sma20, rvol20, taker_buy_share

    Returns:
        Dictionary mapping each feature name to its 1D np.float64 array.
    """
    if not isinstance(arrays, NumericalCandleArrays):
        raise TypeError(f"Expected NumericalCandleArrays, got {type(arrays).__name__}")

    features: Dict[str, np.ndarray] = {}

    # Trend (12)
    features.update(compute_trend_features(arrays.closes))

    # Momentum (8)
    features.update(compute_momentum_features(arrays.closes))

    # Volatility (7)
    features.update(compute_volatility_features(arrays.highs, arrays.lows, arrays.closes))

    # Volume (3)
    features.update(compute_volume_features(arrays.volumes, arrays.taker_buy_base))

    if len(features) != 30:
        raise RuntimeError(f"Expected exactly 30 features, computed {len(features)}")

    return features


__all__ = [
    # Top-level entry
    "compute_all_kernels",
    # Trend
    "calc_ema",
    "calc_price_ema_ratio",
    "calc_ema_spread",
    "calc_ema_slope",
    "compute_trend_features",
    # Momentum
    "calc_rsi",
    "calc_macd",
    "calc_roc",
    "compute_momentum_features",
    # Volatility
    "calc_true_range",
    "calc_atr",
    "calc_natr",
    "calc_bollinger_bands",
    "compute_volatility_features",
    # Volume
    "calc_volume_sma",
    "calc_rvol",
    "calc_taker_buy_share",
    "compute_volume_features",
]
