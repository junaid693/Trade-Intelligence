"""Unit tests for Phase 2.1 Unit 2.2 technical feature calculation kernels.

Covers:
- Mathematical reference verification
- Exact first-valid indices (mathematical availability)
- Edge cases (flat prices, monotonic trends, zero volume, zero variance)
- Per-kernel future-candle temporal invariance (mandatory no-lookahead proof)
- Contiguous segment isolation (gap/reset independence)
- Shape, dtype (np.float64), and input immutability
- Complete 30-feature output mapping
"""

import numpy as np
import pytest

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
from trade_intelligence.features.kernels import compute_all_kernels
from trade_intelligence.features.types import NumericalCandleArrays


# ===========================================================================
# Helper fixtures / synthetic data generators
# ===========================================================================

def _make_mock_arrays(n: int, base_price: float = 100.0) -> NumericalCandleArrays:
    """Generate deterministic, contiguous NumericalCandleArrays of length n."""
    np.random.seed(42)
    # Drift price series
    steps = np.sin(np.linspace(0, 10, n)) * 2.0 + np.linspace(0, 5, n)
    closes = np.round(base_price + steps, 4)
    highs = np.round(closes + np.random.uniform(0.5, 2.0, n), 4)
    lows = np.round(closes - np.random.uniform(0.5, 2.0, n), 4)
    opens = np.round((highs + lows) / 2.0, 4)
    volumes = np.round(np.random.uniform(10.0, 100.0, n), 4)
    quote_volumes = np.round(volumes * closes, 4)
    taker_buy_base = np.round(volumes * np.random.uniform(0.3, 0.7, n), 4)
    taker_buy_quote = np.round(taker_buy_base * closes, 4)
    trades = np.random.randint(5, 50, n).astype(np.int64)

    open_times = np.arange(n, dtype=np.int64) * 3600000
    close_times = open_times + 3599999

    return NumericalCandleArrays(
        symbol="BTCUSDT",
        interval="1h",
        length=n,
        open_times=open_times,
        close_times=close_times,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        quote_volumes=quote_volumes,
        trades=trades,
        taker_buy_base=taker_buy_base,
        taker_buy_quote=taker_buy_quote,
    )


# ===========================================================================
# 1. Trend Kernels Tests
# ===========================================================================

class TestTrendKernels:
    def test_ema_mathematical_reference(self):
        """Hand-calculated EMA reference: prices=[10, 11, 12, 13, 14, 15], period=5."""
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)
        ema = calc_ema(prices, period=5)

        # Indices 0..3 must be NaN
        assert np.all(np.isnan(ema[:4]))
        # Index 4: SMA seed = mean(10, 11, 12, 13, 14) = 12.0
        assert ema[4] == pytest.approx(12.0)
        # Index 5: alpha = 2 / (5 + 1) = 1/3. EMA[5] = (1/3)*15 + (2/3)*12 = 5 + 8 = 13.0
        assert ema[5] == pytest.approx(13.0)

    def test_ema_first_valid_indices(self):
        """First valid index is exactly period - 1 for EMA20 (19), EMA50 (49), EMA200 (199)."""
        n = 250
        prices = np.linspace(100.0, 200.0, n, dtype=np.float64)

        ema20 = calc_ema(prices, 20)
        assert np.all(np.isnan(ema20[:19]))
        assert np.isfinite(ema20[19])

        ema50 = calc_ema(prices, 50)
        assert np.all(np.isnan(ema50[:49]))
        assert np.isfinite(ema50[49])

        ema200 = calc_ema(prices, 200)
        assert np.all(np.isnan(ema200[:199]))
        assert np.isfinite(ema200[199])

    def test_price_ema_ratio(self):
        """Price / EMA ratio matches closes / ema and is NaN when EMA is NaN."""
        closes = np.array([100.0, 105.0, 110.0], dtype=np.float64)
        ema = np.array([np.nan, 100.0, 105.0], dtype=np.float64)
        ratios = calc_price_ema_ratio(closes, ema)

        assert np.isnan(ratios[0])
        assert ratios[1] == pytest.approx(105.0 / 100.0)
        assert ratios[2] == pytest.approx(110.0 / 105.0)

    def test_ema_spread(self):
        """EMA spread is (fast - slow) / slow and NaN where slow is NaN."""
        fast = np.array([np.nan, 110.0, 120.0], dtype=np.float64)
        slow = np.array([np.nan, 100.0, 100.0], dtype=np.float64)
        spread = calc_ema_spread(fast, slow)

        assert np.isnan(spread[0])
        assert spread[1] == pytest.approx((110.0 - 100.0) / 100.0)  # +0.10 (+10%)
        assert spread[2] == pytest.approx((120.0 - 100.0) / 100.0)  # +0.20 (+20%)

    def test_ema_slope_mathematical_reference_and_first_valid(self):
        """EMA slope over lookback k=3: (ema[i] - ema[i-3]) / (3 * ema[i-3])."""
        # EMA first valid at index 4 (length 5)
        ema = np.array([np.nan, np.nan, np.nan, np.nan, 100.0, 103.0, 106.0, 109.0], dtype=np.float64)
        slope = calc_ema_slope(ema, slope_k=3)

        # Index 4 + 3 = 7 is the first index where ema[i-3] is valid
        assert np.all(np.isnan(slope[:7]))
        # At index 7: ema[7]=109.0, ema[4]=100.0 -> slope = (109 - 100) / (3 * 100) = 9 / 300 = 0.03
        assert slope[7] == pytest.approx(0.03)

    def test_ema_slope_first_valid_indices(self):
        """First valid slope indices: slope20 -> 22, slope50 -> 52, slope200 -> 202."""
        n = 250
        prices = np.linspace(100.0, 200.0, n, dtype=np.float64)

        ema20 = calc_ema(prices, 20)
        slope20 = calc_ema_slope(ema20, slope_k=3)
        assert np.all(np.isnan(slope20[:22]))
        assert np.isfinite(slope20[22])

        ema50 = calc_ema(prices, 50)
        slope50 = calc_ema_slope(ema50, slope_k=3)
        assert np.all(np.isnan(slope50[:52]))
        assert np.isfinite(slope50[52])

        ema200 = calc_ema(prices, 200)
        slope200 = calc_ema_slope(ema200, slope_k=3)
        assert np.all(np.isnan(slope200[:202]))
        assert np.isfinite(slope200[202])

    def test_trend_future_candle_invariance(self):
        """MANDATORY: Appending a future candle must not alter historical trend features."""
        n = 220
        prices_a = np.linspace(100.0, 200.0, n, dtype=np.float64)
        prices_b = np.append(prices_a, 215.0)

        feats_a = compute_trend_features(prices_a)
        feats_b = compute_trend_features(prices_b)

        assert len(feats_a) == 12
        for name in feats_a:
            np.testing.assert_allclose(
                feats_a[name],
                feats_b[name][:n],
                rtol=1e-7,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"Future-candle invariance violated for trend feature {name}",
            )


# ===========================================================================
# 2. Momentum Kernels Tests
# ===========================================================================

class TestMomentumKernels:
    def test_rsi_mathematical_reference_monotonic(self):
        """Monotonically increasing series produces RSI = 100; decreasing produces RSI = 0."""
        # Monotonically increasing 20 prices
        up_prices = np.linspace(100.0, 120.0, 20, dtype=np.float64)
        rsi_up = calc_rsi(up_prices, period=14)
        assert np.all(np.isnan(rsi_up[:14]))
        assert np.all(rsi_up[14:] == pytest.approx(100.0))

        # Monotonically decreasing 20 prices
        down_prices = np.linspace(120.0, 100.0, 20, dtype=np.float64)
        rsi_down = calc_rsi(down_prices, period=14)
        assert np.all(np.isnan(rsi_down[:14]))
        assert np.all(rsi_down[14:] == pytest.approx(0.0))

    def test_rsi_flat_prices(self):
        """Completely flat series (zero gains, zero losses) produces neutral RSI = 50.0."""
        flat_prices = np.full(25, 100.0, dtype=np.float64)
        rsi_flat = calc_rsi(flat_prices, period=14)
        assert np.all(np.isnan(rsi_flat[:14]))
        assert np.all(rsi_flat[14:] == pytest.approx(50.0))

    def test_rsi_first_valid_index(self):
        """RSI14 is first valid strictly at index 14 (indices 0..13 are NaN)."""
        prices = np.array([100.0 + i + (i % 2) * 0.5 for i in range(30)], dtype=np.float64)
        rsi = calc_rsi(prices, period=14)
        assert np.all(np.isnan(rsi[:14]))
        assert np.isfinite(rsi[14])

    def test_macd_mathematical_reference(self):
        """MACD line is ema12 - ema26; signal line is 9-period EMA seeded at index 33."""
        prices = np.linspace(100.0, 200.0, 50, dtype=np.float64)
        macd_line, macd_signal, macd_hist, macd_norm, macd_hist_norm = calc_macd(prices)

        # MACD line: first valid at index 25 (26-period EMA seed)
        assert np.all(np.isnan(macd_line[:25]))
        assert np.isfinite(macd_line[25])

        # Signal line and Histogram: first valid at index 33 (25 + 9 - 1)
        assert np.all(np.isnan(macd_signal[:33]))
        assert np.isfinite(macd_signal[33])
        assert np.all(np.isnan(macd_hist[:33]))
        assert np.isfinite(macd_hist[33])

        # Histogram relationship: macd_hist == macd_line - macd_signal
        for i in range(33, 50):
            assert macd_hist[i] == pytest.approx(macd_line[i] - macd_signal[i])

        # Normalized MACD checks (LOCKED): (macd_line / close) * 100
        assert np.all(np.isnan(macd_norm[:25]))
        assert macd_norm[25] == pytest.approx((macd_line[25] / prices[25]) * 100.0)
        assert np.all(np.isnan(macd_hist_norm[:33]))
        assert macd_hist_norm[33] == pytest.approx((macd_hist[33] / prices[33]) * 100.0)

    def test_roc_mathematical_reference_and_first_valid(self):
        """ROC10 first valid at index 10; ROC21 first valid at index 21."""
        prices = np.linspace(100.0, 200.0, 40, dtype=np.float64)

        roc10 = calc_roc(prices, 10)
        assert np.all(np.isnan(roc10[:10]))
        assert np.isfinite(roc10[10])
        # Formula check: ((prices[10] - prices[0]) / prices[0]) * 100
        assert roc10[10] == pytest.approx(((prices[10] - prices[0]) / prices[0]) * 100.0)

        roc21 = calc_roc(prices, 21)
        assert np.all(np.isnan(roc21[:21]))
        assert np.isfinite(roc21[21])
        assert roc21[21] == pytest.approx(((prices[21] - prices[0]) / prices[0]) * 100.0)

    def test_momentum_future_candle_invariance(self):
        """MANDATORY: Appending a future candle must not alter historical momentum features."""
        n = 60
        prices_a = np.array([100.0 + np.sin(i * 0.2) * 5.0 + i * 0.5 for i in range(n)], dtype=np.float64)
        prices_b = np.append(prices_a, 135.0)

        feats_a = compute_momentum_features(prices_a)
        feats_b = compute_momentum_features(prices_b)

        assert len(feats_a) == 8
        for name in feats_a:
            np.testing.assert_allclose(
                feats_a[name],
                feats_b[name][:n],
                rtol=1e-7,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"Future-candle invariance violated for momentum feature {name}",
            )


# ===========================================================================
# 3. Volatility Kernels Tests
# ===========================================================================

class TestVolatilityKernels:
    def test_true_range_reference(self):
        """True Range calculation: TR[0] = high[0] - low[0]; TR[i] uses max(H-L, |H-Cp|, |L-Cp|)."""
        highs = np.array([105.0, 110.0, 108.0], dtype=np.float64)
        lows = np.array([95.0, 100.0, 92.0], dtype=np.float64)
        closes = np.array([100.0, 102.0, 95.0], dtype=np.float64)

        tr = calc_true_range(highs, lows, closes)
        # TR[0] = 105 - 95 = 10.0
        assert tr[0] == pytest.approx(10.0)
        # TR[1] = max(110 - 100 = 10, |110 - 100| = 10, |100 - 100| = 0) = 10.0
        assert tr[1] == pytest.approx(10.0)
        # TR[2] = max(108 - 92 = 16, |108 - 102| = 6, |92 - 102| = 10) = 16.0
        assert tr[2] == pytest.approx(16.0)

    def test_atr_locked_seed_and_first_valid(self):
        """ATR14 is first valid strictly at index 14 (indices 0..13 NaN), seeded with mean(TR[1:15])."""
        n = 20
        highs = np.array([100.0 + i + 2.0 for i in range(n)], dtype=np.float64)
        lows = np.array([100.0 + i - 2.0 for i in range(n)], dtype=np.float64)
        closes = np.array([100.0 + i for i in range(n)], dtype=np.float64)

        tr = calc_true_range(highs, lows, closes)
        atr = calc_atr(highs, lows, closes, period=14)

        # Indices 0..13 must be NaN
        assert np.all(np.isnan(atr[:14]))
        # First valid index is 14: mean(TR[1:15])
        expected_seed = np.mean(tr[1:15])
        assert atr[14] == pytest.approx(expected_seed)

        # Recurrence at index 15: (13 * ATR[14] + TR[15]) / 14
        expected_15 = (13.0 * expected_seed + tr[15]) / 14.0
        assert atr[15] == pytest.approx(expected_15)

    def test_natr_calculation(self):
        """NATR is (ATR / close) * 100."""
        closes = np.array([100.0, 200.0], dtype=np.float64)
        atr = np.array([np.nan, 4.0], dtype=np.float64)
        natr = calc_natr(closes, atr)

        assert np.isnan(natr[0])
        assert natr[1] == pytest.approx((4.0 / 200.0) * 100.0)  # 2.0%

    def test_bollinger_bands_reference_and_first_valid(self):
        """Bollinger Bands first valid at index 19 with ddof=0 population standard deviation."""
        n = 25
        closes = np.array([100.0 + (i % 5) for i in range(n)], dtype=np.float64)
        upper, mid, lower, pct_b, bandwidth = calc_bollinger_bands(closes, period=20, multiplier=2.0, ddof=0)

        # Indices 0..18 must be NaN
        assert np.all(np.isnan(upper[:19]))
        assert np.all(np.isnan(mid[:19]))
        assert np.all(np.isnan(lower[:19]))
        assert np.all(np.isnan(pct_b[:19]))
        assert np.all(np.isnan(bandwidth[:19]))

        # Index 19 checks
        window_19 = closes[:20]
        expected_mean = np.mean(window_19)
        expected_sigma = np.std(window_19, ddof=0)
        assert mid[19] == pytest.approx(expected_mean)
        assert upper[19] == pytest.approx(expected_mean + 2.0 * expected_sigma)
        assert lower[19] == pytest.approx(expected_mean - 2.0 * expected_sigma)

    def test_bollinger_locked_zero_variance_guard(self):
        """LOCKED: When sigma <= 1e-12, pct_b = 0.5 and bandwidth = 0.0."""
        # Exactly flat closes
        closes = np.full(25, 100.0, dtype=np.float64)
        upper, mid, lower, pct_b, bandwidth = calc_bollinger_bands(closes, period=20, multiplier=2.0, ddof=0)

        assert np.all(np.isnan(pct_b[:19]))
        # For indices >= 19, zero variance triggers locked fallback
        assert np.all(pct_b[19:] == pytest.approx(0.5))
        assert np.all(bandwidth[19:] == pytest.approx(0.0))

    def test_volatility_future_candle_invariance(self):
        """MANDATORY: Appending a future candle must not alter historical volatility features."""
        n = 50
        highs_a = np.array([105.0 + i * 0.5 + np.sin(i) * 2.0 for i in range(n)], dtype=np.float64)
        lows_a = np.array([95.0 + i * 0.5 - np.sin(i) * 2.0 for i in range(n)], dtype=np.float64)
        closes_a = np.array([100.0 + i * 0.5 for i in range(n)], dtype=np.float64)

        highs_b = np.append(highs_a, 132.0)
        lows_b = np.append(lows_a, 122.0)
        closes_b = np.append(closes_a, 127.0)

        feats_a = compute_volatility_features(highs_a, lows_a, closes_a)
        feats_b = compute_volatility_features(highs_b, lows_b, closes_b)

        assert len(feats_a) == 7
        for name in feats_a:
            np.testing.assert_allclose(
                feats_a[name],
                feats_b[name][:n],
                rtol=1e-7,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"Future-candle invariance violated for volatility feature {name}",
            )


# ===========================================================================
# 4. Volume Kernels Tests
# ===========================================================================

class TestVolumeKernels:
    def test_volume_sma_first_valid_index(self):
        """Volume SMA20 is first valid at index 19 (indices 0..18 NaN)."""
        volumes = np.full(25, 10.0, dtype=np.float64)
        v_sma = calc_volume_sma(volumes, period=20)
        assert np.all(np.isnan(v_sma[:19]))
        assert v_sma[19] == pytest.approx(10.0)

    def test_rvol_calculation_and_zero_guard(self):
        """RVOL is volume / vol_sma, and defaults to 1.0 when vol_sma == 0."""
        volumes = np.array([10.0, 20.0, 0.0], dtype=np.float64)
        vol_sma = np.array([np.nan, 10.0, 0.0], dtype=np.float64)
        rvol = calc_rvol(volumes, vol_sma)

        assert np.isnan(rvol[0])
        assert rvol[1] == pytest.approx(2.0)
        assert rvol[2] == pytest.approx(1.0)  # Zero SMA returns 1.0 neutral baseline

    def test_taker_buy_share_calculation_and_zero_guard(self):
        """Taker-buy share is taker_buy / volume, and defaults to 0.5 when volume == 0."""
        tb = np.array([6.0, 0.0, 0.0], dtype=np.float64)
        vol = np.array([10.0, 10.0, 0.0], dtype=np.float64)
        share = calc_taker_buy_share(tb, vol)

        assert share[0] == pytest.approx(0.60)
        assert share[1] == pytest.approx(0.0)
        assert share[2] == pytest.approx(0.5)  # Zero volume returns 0.5 neutral participation

    def test_volume_future_candle_invariance(self):
        """MANDATORY: Appending a future candle must not alter historical volume features."""
        n = 30
        vol_a = np.array([10.0 + (i % 7) * 5.0 for i in range(n)], dtype=np.float64)
        tb_a = np.array([v * 0.45 for v in vol_a], dtype=np.float64)

        vol_b = np.append(vol_a, 50.0)
        tb_b = np.append(tb_a, 25.0)

        feats_a = compute_volume_features(vol_a, tb_a)
        feats_b = compute_volume_features(vol_b, tb_b)

        assert len(feats_a) == 3
        for name in feats_a:
            np.testing.assert_allclose(
                feats_a[name],
                feats_b[name][:n],
                rtol=1e-7,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"Future-candle invariance violated for volume feature {name}",
            )


# ===========================================================================
# 5. Segment Gap & Reset Independence Tests
# ===========================================================================

class TestSegmentIndependence:
    def test_segment_reset_isolation(self):
        """Two separate segments compute independent indicators without state leakage."""
        # Segment 1: flat 100
        p1 = np.full(30, 100.0, dtype=np.float64)
        ema_1 = calc_ema(p1, 20)

        # Segment 2: flat 200
        p2 = np.full(30, 200.0, dtype=np.float64)
        ema_2 = calc_ema(p2, 20)

        # Each segment must produce its own SMA seed at index 19
        assert ema_1[19] == pytest.approx(100.0)
        assert ema_2[19] == pytest.approx(200.0)
        # Indices 0..18 of segment 2 must be NaN, proving zero bleed from segment 1
        assert np.all(np.isnan(ema_2[:19]))


# ===========================================================================
# 6. Combined 30-Feature Compute & Immutability Tests
# ===========================================================================

class TestComputeAllKernels:
    def test_exactly_30_features_computed(self):
        """compute_all_kernels computes exactly the 30 locked features with matching length and dtype float64."""
        arrays = _make_mock_arrays(n=250)
        features = compute_all_kernels(arrays)

        assert len(features) == 30

        expected_features = {
            # Trend (12)
            "ema20", "ema50", "ema200",
            "ratio_close_ema20", "ratio_close_ema50", "ratio_close_ema200",
            "spread_ema20_50", "spread_ema50_200", "spread_ema20_200",
            "slope_ema20", "slope_ema50", "slope_ema200",
            # Momentum (8)
            "rsi14", "macd_line", "macd_signal", "macd_hist",
            "macd_norm", "macd_hist_norm", "roc10", "roc21",
            # Volatility (7)
            "atr14", "natr14", "bb_upper", "bb_mid", "bb_lower",
            "bb_pct_b", "bb_bandwidth",
            # Volume (3)
            "vol_sma20", "rvol20", "taker_buy_share",
        }
        assert set(features.keys()) == expected_features

        for name, arr in features.items():
            assert isinstance(arr, np.ndarray), f"{name} is not a numpy array"
            assert arr.ndim == 1, f"{name} ndim={arr.ndim} != 1"
            assert arr.dtype == np.float64, f"{name} dtype={arr.dtype} != float64"
            assert len(arr) == 250, f"{name} length {len(arr)} != 250"

    def test_input_immutability(self):
        """Kernels must never mutate input arrays."""
        arrays = _make_mock_arrays(n=50)
        closes_copy = arrays.closes.copy()
        highs_copy = arrays.highs.copy()
        lows_copy = arrays.lows.copy()
        vols_copy = arrays.volumes.copy()

        _ = compute_all_kernels(arrays)

        np.testing.assert_array_equal(arrays.closes, closes_copy)
        np.testing.assert_array_equal(arrays.highs, highs_copy)
        np.testing.assert_array_equal(arrays.lows, lows_copy)
        np.testing.assert_array_equal(arrays.volumes, vols_copy)

    def test_short_sequence_handling(self):
        """Sequences shorter than required lookbacks safely yield NaNs without crashing."""
        arrays = _make_mock_arrays(n=5)
        features = compute_all_kernels(arrays)

        assert len(features) == 30
        for name, arr in features.items():
            assert len(arr) == 5
            # taker_buy_share is valid from index 0
            if name != "taker_buy_share":
                assert np.all(np.isnan(arr)), f"Expected all NaN for {name} with n=5"
            else:
                assert np.all(np.isfinite(arr)), "taker_buy_share should be finite"
