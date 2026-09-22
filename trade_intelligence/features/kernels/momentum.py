"""Momentum feature calculation kernels (RSI, MACD, Normalized MACD, ROC)."""

from typing import Dict, Tuple
import numpy as np

from trade_intelligence.features.kernels.trend import calc_ema


def calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index (RSI) using Wilder's smoothing method.

    Mathematical specification:
    - delta[i] = closes[i] - closes[i - 1] for i >= 1
    - gain[i] = max(delta[i], 0.0)
    - loss[i] = max(-delta[i], 0.0)
    - Seed at index period using indices 1..period:
        avg_gain[period] = mean(gain[1 : period + 1])
        avg_loss[period] = mean(loss[1 : period + 1])
    - For i > period:
        avg_gain[i] = ((period - 1) * avg_gain[i - 1] + gain[i]) / period
        avg_loss[i] = ((period - 1) * avg_loss[i - 1] + loss[i]) / period
    - RSI[i] = 100.0 * avg_gain[i] / (avg_gain[i] + avg_loss[i])
      Special cases:
        avg_gain == 0 and avg_loss == 0 -> 50.0
        avg_loss == 0 and avg_gain > 0 -> 100.0
    - For i < period: output is np.nan.

    Args:
        closes: 1D array of close prices (np.float64).
        period: Positive integer period (locked default = 14).

    Returns:
        1D np.float64 array of RSI values.
    """
    if not isinstance(closes, np.ndarray):
        raise TypeError(f"closes must be a numpy ndarray, got {type(closes).__name__}")
    if closes.ndim != 1:
        raise ValueError(f"closes must be 1-dimensional, got ndim={closes.ndim}")
    if not isinstance(period, int) or isinstance(period, bool) or period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")

    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    # Requires at least period + 1 candles (e.g. 15 candles for period=14) to compute 14 deltas
    if n < period + 1:
        return out

    # Compute deltas
    deltas = np.diff(closes)  # length n - 1, index k corresponds to delta between candle k+1 and k
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)

    # Initial seed at index period (uses deltas from candle 1 to period, indices 0 to period-1 of deltas)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    if avg_gain == 0.0 and avg_loss == 0.0:
        out[period] = 50.0
    elif avg_loss == 0.0:
        out[period] = 100.0
    else:
        out[period] = 100.0 * avg_gain / (avg_gain + avg_loss)

    # Wilder recurrence
    weight = float(period - 1)
    period_f = float(period)

    for i in range(period + 1, n):
        d_idx = i - 1  # delta index corresponding to candle i
        avg_gain = (weight * avg_gain + gains[d_idx]) / period_f
        avg_loss = (weight * avg_loss + losses[d_idx]) / period_f

        if avg_gain == 0.0 and avg_loss == 0.0:
            out[i] = 50.0
        elif avg_loss == 0.0:
            out[i] = 100.0
        else:
            out[i] = 100.0 * avg_gain / (avg_gain + avg_loss)

    return out


def calc_macd(
    closes: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate MACD Line, Signal Line, Histogram, and Normalized MACD metrics.

    Mathematical specification:
    - fast_ema = calc_ema(closes, fast_period) (e.g. 12)
    - slow_ema = calc_ema(closes, slow_period) (e.g. 26)
    - macd_line = fast_ema - slow_ema (first valid at index slow_period - 1, e.g. 25)
    - signal line: 9-period EMA of macd_line
      Seed at index 33: mean(macd_line[25:34])
      For i > 33: signal[i] = 0.2 * macd_line[i] + 0.8 * signal[i - 1]
    - macd_hist = macd_line - signal (first valid at index 33)
    - macd_norm = (macd_line / close) * 100
    - macd_hist_norm = (macd_hist / close) * 100

    Returns:
        Tuple of (macd_line, macd_signal, macd_hist, macd_norm, macd_hist_norm).
    """
    n = len(closes)
    macd_line = np.full(n, np.nan, dtype=np.float64)
    macd_signal = np.full(n, np.nan, dtype=np.float64)
    macd_hist = np.full(n, np.nan, dtype=np.float64)
    macd_norm = np.full(n, np.nan, dtype=np.float64)
    macd_hist_norm = np.full(n, np.nan, dtype=np.float64)

    fast_ema = calc_ema(closes, fast_period)
    slow_ema = calc_ema(closes, slow_period)

    # MACD line valid where both EMAs are valid
    valid_macd_mask = np.isfinite(fast_ema) & np.isfinite(slow_ema)
    macd_line[valid_macd_mask] = fast_ema[valid_macd_mask] - slow_ema[valid_macd_mask]

    # Normalized MACD line
    valid_norm_mask = valid_macd_mask & np.isfinite(closes) & (closes > 0.0)
    macd_norm[valid_norm_mask] = (macd_line[valid_norm_mask] / closes[valid_norm_mask]) * 100.0

    # Signal line requires signal_period consecutive valid MACD values
    # In a contiguous series, slow_ema first valid index is slow_period - 1 (e.g. 25)
    # Signal seed uses indices [25 : 25 + 9] = indices 25..33
    start_idx = slow_period - 1
    signal_seed_end = start_idx + signal_period  # e.g. 25 + 9 = 34 (indices 25..33)
    signal_first_valid = signal_seed_end - 1     # e.g. 33

    if n >= signal_seed_end and np.all(np.isfinite(macd_line[start_idx:signal_seed_end])):
        sig_seed = np.mean(macd_line[start_idx:signal_seed_end])
        macd_signal[signal_first_valid] = sig_seed
        macd_hist[signal_first_valid] = macd_line[signal_first_valid] - sig_seed
        if np.isfinite(closes[signal_first_valid]) and closes[signal_first_valid] > 0.0:
            macd_hist_norm[signal_first_valid] = (macd_hist[signal_first_valid] / closes[signal_first_valid]) * 100.0

        alpha = 2.0 / (signal_period + 1.0)  # 2 / 10 = 0.2
        one_minus_alpha = 1.0 - alpha         # 0.8

        curr_sig = sig_seed
        for i in range(signal_first_valid + 1, n):
            if np.isfinite(macd_line[i]):
                curr_sig = alpha * macd_line[i] + one_minus_alpha * curr_sig
                macd_signal[i] = curr_sig
                h = macd_line[i] - curr_sig
                macd_hist[i] = h
                if np.isfinite(closes[i]) and closes[i] > 0.0:
                    macd_hist_norm[i] = (h / closes[i]) * 100.0
            else:
                # If an invalid MACD line occurs, break signal calculation
                break

    return macd_line, macd_signal, macd_hist, macd_norm, macd_hist_norm


def calc_roc(closes: np.ndarray, period: int) -> np.ndarray:
    """Calculate Rate of Change (ROC): ((close[i] - close[i - period]) / close[i - period]) * 100.

    First valid index is period (requires period + 1 candles, indices 0..period).
    Earlier indices return np.nan.
    """
    if not isinstance(period, int) or isinstance(period, bool) or period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")

    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return out

    for i in range(period, n):
        past_val = closes[i - period]
        curr_val = closes[i]
        if np.isfinite(past_val) and past_val > 0.0 and np.isfinite(curr_val):
            out[i] = ((curr_val - past_val) / past_val) * 100.0

    return out


def compute_momentum_features(closes: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute all 8 momentum features on a contiguous 1D array of close prices.

    Features:
    - rsi14
    - macd_line, macd_signal, macd_hist
    - macd_norm, macd_hist_norm
    - roc10, roc21
    """
    rsi14 = calc_rsi(closes, period=14)
    macd_line, macd_signal, macd_hist, macd_norm, macd_hist_norm = calc_macd(
        closes, fast_period=12, slow_period=26, signal_period=9
    )
    roc10 = calc_roc(closes, period=10)
    roc21 = calc_roc(closes, period=21)

    return {
        "rsi14": rsi14,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "macd_norm": macd_norm,
        "macd_hist_norm": macd_hist_norm,
        "roc10": roc10,
        "roc21": roc21,
    }
