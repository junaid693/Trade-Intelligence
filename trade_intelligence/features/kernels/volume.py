"""Volume feature calculation kernels (Volume SMA20, RVOL20, Taker-Buy Volume Share)."""

from typing import Dict
import numpy as np


def calc_volume_sma(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate Simple Moving Average (SMA) of volume.

    Mathematical specification:
    - vol_sma[i] = mean(volumes[i - period + 1 : i + 1]) for i >= period - 1
    - For i < period - 1: output is np.nan.

    Args:
        volumes: 1D array of volume values (np.float64).
        period: Positive integer period (locked default = 20).

    Returns:
        1D np.float64 array of Volume SMA values.
    """
    if not isinstance(period, int) or isinstance(period, bool) or period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")

    n = len(volumes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out

    for i in range(period - 1, n):
        out[i] = float(np.mean(volumes[i - period + 1 : i + 1]))

    return out


def calc_rvol(volumes: np.ndarray, vol_sma: np.ndarray) -> np.ndarray:
    """Calculate Relative Volume (RVOL): volume / vol_sma.

    Mathematical specification:
    - If vol_sma == 0: return 1.0 (neutral baseline).
    - If vol_sma is np.nan: return np.nan.
    - Otherwise: volume / vol_sma.

    Returns:
        1D np.float64 array of RVOL values.
    """
    if len(volumes) != len(vol_sma):
        raise ValueError(f"Array length mismatch: volumes ({len(volumes)}) != vol_sma ({len(vol_sma)})")

    n = len(volumes)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        s = vol_sma[i]
        v = volumes[i]
        if np.isnan(s):
            out[i] = np.nan
        elif s == 0.0:
            out[i] = 1.0
        elif np.isfinite(s) and np.isfinite(v):
            out[i] = v / s

    return out


def calc_taker_buy_share(taker_buy_base: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Calculate Taker-Buy Volume Share: taker_buy_base_volume / total_volume.

    Mathematical specification:
    - If volume == 0: return 0.5 (neutral participation).
    - If volume is np.nan or taker_buy_base is np.nan: return np.nan.
    - Otherwise: taker_buy_base / volume.
    - Output is bounded in [0.0, 1.0]. Valid from index 0.

    Returns:
        1D np.float64 array of Taker-Buy Volume Share values.
    """
    if len(taker_buy_base) != len(volumes):
        raise ValueError(
            f"Array length mismatch: taker_buy_base ({len(taker_buy_base)}) != volumes ({len(volumes)})"
        )

    n = len(volumes)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        v = volumes[i]
        tb = taker_buy_base[i]
        if np.isnan(v) or np.isnan(tb):
            out[i] = np.nan
        elif v == 0.0:
            out[i] = 0.5
        elif np.isfinite(v) and np.isfinite(tb) and v > 0.0:
            # Bound in [0.0, 1.0]
            share = tb / v
            out[i] = max(0.0, min(1.0, share))

    return out


def compute_volume_features(
    volumes: np.ndarray,
    taker_buy_base: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute all 3 volume features on a contiguous 1D slice.

    Features:
    - vol_sma20
    - rvol20
    - taker_buy_share
    """
    vol_sma20 = calc_volume_sma(volumes, period=20)
    rvol20 = calc_rvol(volumes, vol_sma20)
    taker_buy_share = calc_taker_buy_share(taker_buy_base, volumes)

    return {
        "vol_sma20": vol_sma20,
        "rvol20": rvol20,
        "taker_buy_share": taker_buy_share,
    }
