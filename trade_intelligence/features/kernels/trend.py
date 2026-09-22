"""Trend feature calculation kernels (EMAs, Ratios, Spreads, Slope)."""

from typing import Dict, Optional
import numpy as np


def calc_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average (EMA) seeded with an arithmetic mean (SMA).

    Mathematical specification:
    - alpha = 2 / (period + 1)
    - EMA[period - 1] = mean(prices[0:period])
    - For i >= period: EMA[i] = alpha * prices[i] + (1 - alpha) * EMA[i - 1]
    - For i < period - 1: EMA[i] = np.nan

    Args:
        prices: 1D array of prices (np.float64).
        period: Positive integer period (e.g. 20, 50, 200).

    Returns:
        1D np.float64 array of EMA values, matching len(prices).
    """
    if not isinstance(prices, np.ndarray):
        raise TypeError(f"prices must be a numpy ndarray, got {type(prices).__name__}")
    if prices.ndim != 1:
        raise ValueError(f"prices must be 1-dimensional, got ndim={prices.ndim}")
    if not isinstance(period, int) or isinstance(period, bool) or period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")

    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out

    # SMA seed at index period - 1
    seed = np.mean(prices[:period])
    out[period - 1] = seed

    alpha = 2.0 / (period + 1.0)
    one_minus_alpha = 1.0 - alpha

    # Sequential recursive evaluation
    curr_ema = seed
    for i in range(period, n):
        curr_ema = alpha * prices[i] + one_minus_alpha * curr_ema
        out[i] = curr_ema

    return out


def calc_price_ema_ratio(closes: np.ndarray, ema: np.ndarray) -> np.ndarray:
    """Calculate ratio of close price to EMA: close / EMA.

    Returns np.nan where EMA is unavailable (np.nan) or non-positive (<= 0).
    """
    if len(closes) != len(ema):
        raise ValueError(f"Array length mismatch: closes ({len(closes)}) != ema ({len(ema)})")

    out = np.full(len(closes), np.nan, dtype=np.float64)
    valid_mask = np.isfinite(ema) & (ema > 0.0) & np.isfinite(closes)
    out[valid_mask] = closes[valid_mask] / ema[valid_mask]
    return out


def calc_ema_spread(ema_fast: np.ndarray, ema_slow: np.ndarray) -> np.ndarray:
    """Calculate normalized percentage distance between EMAs: (fast - slow) / slow.

    Returns np.nan where either EMA is unavailable (np.nan) or slow is non-positive (<= 0).
    """
    if len(ema_fast) != len(ema_slow):
        raise ValueError(f"Array length mismatch: fast ({len(ema_fast)}) != slow ({len(ema_slow)})")

    out = np.full(len(ema_fast), np.nan, dtype=np.float64)
    valid_mask = np.isfinite(ema_fast) & np.isfinite(ema_slow) & (ema_slow > 0.0)
    out[valid_mask] = (ema_fast[valid_mask] - ema_slow[valid_mask]) / ema_slow[valid_mask]
    return out


def calc_ema_slope(ema: np.ndarray, slope_k: int = 3) -> np.ndarray:
    """Calculate normalized EMA slope over lookback k: (ema[i] - ema[i - k]) / (k * ema[i - k]).

    Args:
        ema: 1D array of EMA values (np.float64).
        slope_k: Positive integer lookback delta (locked k = 3).

    Returns:
        1D np.float64 array of normalized slope values.
    """
    if not isinstance(slope_k, int) or isinstance(slope_k, bool) or slope_k < 1:
        raise ValueError(f"slope_k must be a positive integer, got {slope_k!r}")

    n = len(ema)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= slope_k:
        return out

    # For each index i >= slope_k, evaluate normalized slope
    denominator_k = float(slope_k)
    for i in range(slope_k, n):
        curr_val = ema[i]
        prev_val = ema[i - slope_k]
        if np.isfinite(curr_val) and np.isfinite(prev_val) and prev_val > 0.0:
            out[i] = (curr_val - prev_val) / (denominator_k * prev_val)

    return out


def compute_trend_features(closes: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute all 12 trend features on a contiguous 1D array of close prices.

    Features:
    - ema20, ema50, ema200
    - ratio_close_ema20, ratio_close_ema50, ratio_close_ema200
    - spread_ema20_50, spread_ema50_200, spread_ema20_200
    - slope_ema20, slope_ema50, slope_ema200 (k=3)
    """
    ema20 = calc_ema(closes, 20)
    ema50 = calc_ema(closes, 50)
    ema200 = calc_ema(closes, 200)

    ratio20 = calc_price_ema_ratio(closes, ema20)
    ratio50 = calc_price_ema_ratio(closes, ema50)
    ratio200 = calc_price_ema_ratio(closes, ema200)

    spread_20_50 = calc_ema_spread(ema20, ema50)
    spread_50_200 = calc_ema_spread(ema50, ema200)
    spread_20_200 = calc_ema_spread(ema20, ema200)

    slope20 = calc_ema_slope(ema20, slope_k=3)
    slope50 = calc_ema_slope(ema50, slope_k=3)
    slope200 = calc_ema_slope(ema200, slope_k=3)

    return {
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "ratio_close_ema20": ratio20,
        "ratio_close_ema50": ratio50,
        "ratio_close_ema200": ratio200,
        "spread_ema20_50": spread_20_50,
        "spread_ema50_200": spread_50_200,
        "spread_ema20_200": spread_20_200,
        "slope_ema20": slope20,
        "slope_ema50": slope50,
        "slope_ema200": slope200,
    }
