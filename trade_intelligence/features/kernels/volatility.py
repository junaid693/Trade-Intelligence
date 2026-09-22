"""Volatility feature calculation kernels (True Range, ATR14, NATR, Bollinger Bands)."""

from typing import Dict, Tuple
import numpy as np


def calc_true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Calculate True Range (TR) for each candle in a contiguous segment.

    Mathematical specification:
    - At index 0: TR[0] = highs[0] - lows[0]
    - For i >= 1: TR[i] = max(
        highs[i] - lows[i],
        abs(highs[i] - closes[i - 1]),
        abs(lows[i] - closes[i - 1])
      )

    Args:
        highs: 1D array of high prices (np.float64).
        lows: 1D array of low prices (np.float64).
        closes: 1D array of close prices (np.float64).

    Returns:
        1D np.float64 array of True Range values.
    """
    n = len(highs)
    if len(lows) != n or len(closes) != n:
        raise ValueError(
            f"Array length mismatch: highs ({n}), lows ({len(lows)}), closes ({len(closes)})"
        )

    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    # Index 0: high - low
    out[0] = highs[0] - lows[0]
    if n == 1:
        return out

    # Indices >= 1
    hl = highs[1:] - lows[1:]
    hc = np.abs(highs[1:] - closes[:-1])
    lc = np.abs(lows[1:] - closes[:-1])
    out[1:] = np.maximum(hl, np.maximum(hc, lc))

    return out


def calc_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Calculate Average True Range (ATR) using Wilder's exact smoothing method.

    Mathematical specification (LOCKED):
    - TR is calculated via calc_true_range.
    - Initial seed at index period (e.g. 14):
        ATR[period] = mean(TR[1 : period + 1])
        (Mean of TR[1] through TR[period]; TR[0] is NOT used in the ATR seed).
    - For i > period:
        ATR[i] = ((period - 1) * ATR[i - 1] + TR[i]) / period
    - For i < period: output is np.nan.

    Args:
        highs: 1D array of high prices (np.float64).
        lows: 1D array of low prices (np.float64).
        closes: 1D array of close prices (np.float64).
        period: Positive integer period (locked default = 14).

    Returns:
        1D np.float64 array of ATR values.
    """
    if not isinstance(period, int) or isinstance(period, bool) or period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")

    n = len(highs)
    out = np.full(n, np.nan, dtype=np.float64)
    # Requires at least period + 1 candles (e.g. 15 candles for period=14)
    if n < period + 1:
        return out

    tr = calc_true_range(highs, lows, closes)

    # Initial seed at index period: mean(TR[1 : period + 1])
    # Note: TR[0] is excluded from the seed
    seed = float(np.mean(tr[1 : period + 1]))
    out[period] = seed

    weight = float(period - 1)
    period_f = float(period)

    curr_atr = seed
    for i in range(period + 1, n):
        curr_atr = (weight * curr_atr + tr[i]) / period_f
        out[i] = curr_atr

    return out


def calc_natr(closes: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """Calculate Normalized Average True Range (NATR): (atr / close) * 100.

    Returns np.nan where ATR is unavailable or close is non-positive (<= 0).
    """
    if len(closes) != len(atr):
        raise ValueError(f"Array length mismatch: closes ({len(closes)}) != atr ({len(atr)})")

    out = np.full(len(closes), np.nan, dtype=np.float64)
    valid_mask = np.isfinite(atr) & np.isfinite(closes) & (closes > 0.0)
    out[valid_mask] = (atr[valid_mask] / closes[valid_mask]) * 100.0
    return out


def calc_bollinger_bands(
    closes: np.ndarray,
    period: int = 20,
    multiplier: float = 2.0,
    ddof: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands, %B, and Bandwidth.

    Mathematical specification:
    - Middle Band: period-bar SMA of closes
    - Standard deviation: population standard deviation (ddof = 0)
    - Upper Band = bb_mid + multiplier * sigma
    - Lower Band = bb_mid - multiplier * sigma
    - LOCKED zero-variance guard:
        If sigma <= 1e-12:
            bb_pct_b = 0.5
            bb_bandwidth = 0.0
        Otherwise:
            bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower)
            bb_bandwidth = (bb_upper - bb_lower) / bb_mid
    - For i < period - 1: all outputs are np.nan.

    Returns:
        Tuple of (bb_upper, bb_mid, bb_lower, bb_pct_b, bb_bandwidth).
    """
    if not isinstance(period, int) or isinstance(period, bool) or period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")
    if not isinstance(multiplier, (int, float)) or multiplier <= 0.0:
        raise ValueError(f"multiplier must be a positive number, got {multiplier!r}")
    if ddof not in (0, 1):
        raise ValueError(f"ddof must be 0 or 1, got {ddof!r}")

    n = len(closes)
    bb_upper = np.full(n, np.nan, dtype=np.float64)
    bb_mid = np.full(n, np.nan, dtype=np.float64)
    bb_lower = np.full(n, np.nan, dtype=np.float64)
    bb_pct_b = np.full(n, np.nan, dtype=np.float64)
    bb_bandwidth = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return bb_upper, bb_mid, bb_lower, bb_pct_b, bb_bandwidth

    mult = float(multiplier)

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        mean_val = float(np.mean(window))
        # Population standard deviation with ddof=0
        sigma = float(np.std(window, ddof=ddof))

        upper = mean_val + mult * sigma
        lower = mean_val - mult * sigma

        bb_mid[i] = mean_val
        bb_upper[i] = upper
        bb_lower[i] = lower

        # LOCKED zero-variance guard: sigma <= 1e-12
        if sigma <= 1e-12:
            bb_pct_b[i] = 0.5
            bb_bandwidth[i] = 0.0
        else:
            band_width_diff = upper - lower
            bb_pct_b[i] = (closes[i] - lower) / band_width_diff
            if mean_val > 0.0:
                bb_bandwidth[i] = band_width_diff / mean_val
            else:
                bb_bandwidth[i] = 0.0

    return bb_upper, bb_mid, bb_lower, bb_pct_b, bb_bandwidth


def compute_volatility_features(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute all 7 volatility features on a contiguous 1D candle slice.

    Features:
    - atr14
    - natr14
    - bb_upper, bb_mid, bb_lower
    - bb_pct_b, bb_bandwidth
    """
    atr14 = calc_atr(highs, lows, closes, period=14)
    natr14 = calc_natr(closes, atr14)
    bb_upper, bb_mid, bb_lower, bb_pct_b, bb_bandwidth = calc_bollinger_bands(
        closes, period=20, multiplier=2.0, ddof=0
    )

    return {
        "atr14": atr14,
        "natr14": natr14,
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "bb_pct_b": bb_pct_b,
        "bb_bandwidth": bb_bandwidth,
    }
