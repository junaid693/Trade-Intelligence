"""Unit 2.3 tests — Feature Assembly + FeatureMatrix Construction.

Validates shape, 30-feature canonical ordering, kernel integration, dtype,
timestamp alignment, warm-up metadata, gap/segment isolation,
future-candle invariance, immutability, determinism, and invalid input rejection.
"""

import time
from decimal import Decimal
from typing import Dict, List

import numpy as np
import pytest

from trade_intelligence.features.assembly import (
    _build_gap_boundary_mask,
    _build_warmup_mask,
    assemble_feature_matrix,
    assemble_features_for_segment,
    assemble_features_from_candles,
    assemble_features_from_segments,
)
from trade_intelligence.features.exceptions import FeatureAssemblyError
from trade_intelligence.features.types import (
    CANONICAL_FEATURE_NAMES,
    FEATURE_FIRST_VALID_INDICES,
    FEATURE_WARMUP_THRESHOLDS,
    FeatureCandle,
    FeatureMatrix,
    FeatureMetadata,
    NumericalCandleArrays,
    TechnicalFeaturesSpec,
)
from trade_intelligence.features.validation import (
    convert_candles_to_arrays,
    partition_contiguous_segments,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

HOUR_MS = 3_600_000


def _make_arrays(n: int, symbol: str = "BTCUSDT", interval: str = "1h") -> NumericalCandleArrays:
    """Create a deterministic NumericalCandleArrays of length n."""
    rng = np.random.RandomState(42)
    base = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    opens = base.copy()
    highs = base + rng.uniform(0.5, 2.0, n)
    lows = base - rng.uniform(0.5, 2.0, n)
    closes = base + rng.randn(n) * 0.3
    volumes = rng.uniform(10.0, 1000.0, n)
    taker_buy_base = volumes * rng.uniform(0.3, 0.7, n)

    return NumericalCandleArrays(
        symbol=symbol,
        interval=interval,
        length=n,
        open_times=np.arange(n, dtype=np.int64) * HOUR_MS,
        close_times=np.arange(n, dtype=np.int64) * HOUR_MS + HOUR_MS - 1,
        opens=opens.astype(np.float64),
        highs=highs.astype(np.float64),
        lows=lows.astype(np.float64),
        closes=closes.astype(np.float64),
        volumes=volumes.astype(np.float64),
        quote_volumes=(volumes * closes).astype(np.float64),
        trades=rng.randint(1, 100, n).astype(np.int64),
        taker_buy_base=taker_buy_base.astype(np.float64),
        taker_buy_quote=(taker_buy_base * closes).astype(np.float64),
    )


def _make_candles(n: int, start_ms: int = 0, gap_at: int = -1) -> List[FeatureCandle]:
    """Create a list of n deterministic FeatureCandles.

    If gap_at > 0, an interior gap of 2 intervals is inserted at that index.
    """
    rng = np.random.RandomState(42)
    candles: List[FeatureCandle] = []
    ot = start_ms
    for i in range(n):
        if i == gap_at and i > 0:
            ot += HOUR_MS  # skip one interval to create a gap
        base = Decimal("100") + Decimal(str(round(float(np.cumsum(rng.randn(1))[0]) * 0.5, 4)))
        if base <= 0:
            base = Decimal("1")
        high_add = Decimal(str(round(float(rng.uniform(0.5, 2.0)), 4)))
        low_sub = Decimal(str(round(float(rng.uniform(0.5, 2.0)), 4)))
        close_off = Decimal(str(round(float(rng.randn() * 0.3), 4)))

        op = base
        hp = base + high_add
        lp = base - low_sub
        cp = base + close_off

        # Enforce OHLC invariants
        if lp <= 0:
            lp = Decimal("0.0001")
        if hp < op:
            hp = op + Decimal("0.01")
        if hp < cp:
            hp = cp + Decimal("0.01")
        if lp > op:
            lp = op - Decimal("0.01")
            if lp <= 0:
                lp = Decimal("0.0001")
        if lp > cp:
            lp = cp - Decimal("0.01")
            if lp <= 0:
                lp = Decimal("0.0001")
        if cp <= 0:
            cp = Decimal("0.0001")
            lp = min(lp, cp)

        vol = Decimal(str(round(float(rng.uniform(10, 1000)), 4)))
        tb = vol * Decimal(str(round(float(rng.uniform(0.3, 0.7)), 4)))
        qav = vol * cp
        tq = tb * cp

        candles.append(FeatureCandle(
            open_time=ot,
            close_time=ot + HOUR_MS - 1,
            open_price=op,
            high_price=hp,
            low_price=lp,
            close_price=cp,
            volume=vol,
            quote_asset_volume=qav,
            number_of_trades=int(rng.randint(1, 100)),
            taker_buy_base_volume=tb,
            taker_buy_quote_volume=tq,
        ))
        ot += HOUR_MS

    return candles


# ===========================================================================
# 1. CANONICAL CONSTANTS INTEGRITY
# ===========================================================================


class TestCanonicalConstants:
    """Verify canonical feature name tuple and metadata dictionaries."""

    def test_canonical_feature_count(self):
        assert len(CANONICAL_FEATURE_NAMES) == 30

    def test_canonical_unique(self):
        assert len(set(CANONICAL_FEATURE_NAMES)) == 30

    def test_family_ordering(self):
        """Verify Trend(12), Momentum(8), Volatility(7), Volume(3) ordering."""
        trend = CANONICAL_FEATURE_NAMES[:12]
        momentum = CANONICAL_FEATURE_NAMES[12:20]
        volatility = CANONICAL_FEATURE_NAMES[20:27]
        volume = CANONICAL_FEATURE_NAMES[27:30]

        assert len(trend) == 12
        assert len(momentum) == 8
        assert len(volatility) == 7
        assert len(volume) == 3

    def test_first_valid_keys_match(self):
        assert set(FEATURE_FIRST_VALID_INDICES.keys()) == set(CANONICAL_FEATURE_NAMES)

    def test_warmup_keys_match(self):
        assert set(FEATURE_WARMUP_THRESHOLDS.keys()) == set(CANONICAL_FEATURE_NAMES)

    def test_warmup_gte_first_valid(self):
        """Warmup threshold >= first_valid_index for every feature."""
        for name in CANONICAL_FEATURE_NAMES:
            assert FEATURE_WARMUP_THRESHOLDS[name] >= FEATURE_FIRST_VALID_INDICES[name], (
                f"{name}: warmup {FEATURE_WARMUP_THRESHOLDS[name]} < "
                f"first_valid {FEATURE_FIRST_VALID_INDICES[name]}"
            )


# ===========================================================================
# 2. TechnicalFeaturesSpec
# ===========================================================================


class TestTechnicalFeaturesSpec:
    """Verify the unified specification container."""

    def test_defaults_produce_stable_hash(self):
        s1 = TechnicalFeaturesSpec()
        s2 = TechnicalFeaturesSpec()
        assert s1.spec_hash == s2.spec_hash
        assert len(s1.spec_hash) == 64

    def test_family_specs_contain_four_families(self):
        spec = TechnicalFeaturesSpec()
        families = spec.family_specs()
        assert set(families.keys()) == {"trend", "momentum", "volatility", "volume"}

    def test_hash_changes_with_parameters(self):
        from trade_intelligence.features.types import TrendSpec
        s1 = TechnicalFeaturesSpec()
        s2 = TechnicalFeaturesSpec(trend=TrendSpec(slope_k=5))
        assert s1.spec_hash != s2.spec_hash


# ===========================================================================
# 3. SINGLE SEGMENT ASSEMBLY
# ===========================================================================


class TestAssembleSingleSegment:
    """Test assemble_feature_matrix with a single contiguous segment."""

    @pytest.fixture
    def large_arrays(self) -> NumericalCandleArrays:
        """700 bars: enough for all features including ema200 warm-up."""
        return _make_arrays(700)

    @pytest.fixture
    def matrix(self, large_arrays: NumericalCandleArrays) -> FeatureMatrix:
        return assemble_feature_matrix(large_arrays)

    def test_shape(self, matrix: FeatureMatrix, large_arrays: NumericalCandleArrays):
        n = large_arrays.length
        assert len(matrix) == n
        assert len(matrix.features) == 30

    def test_canonical_ordering(self, matrix: FeatureMatrix):
        actual_order = list(matrix.features.keys())
        expected_order = list(CANONICAL_FEATURE_NAMES)
        assert actual_order == expected_order

    def test_all_dtypes_float64(self, matrix: FeatureMatrix):
        for name, arr in matrix.features.items():
            assert arr.dtype == np.float64, f"{name} has dtype {arr.dtype}"

    def test_all_arrays_1d(self, matrix: FeatureMatrix, large_arrays: NumericalCandleArrays):
        for name, arr in matrix.features.items():
            assert arr.ndim == 1
            assert len(arr) == large_arrays.length

    def test_timestamp_alignment(self, matrix: FeatureMatrix, large_arrays: NumericalCandleArrays):
        np.testing.assert_array_equal(matrix.open_times, large_arrays.open_times)
        np.testing.assert_array_equal(matrix.close_times, large_arrays.close_times)

    def test_timestamp_ownership_immutability(self):
        """Mutating source NumericalCandleArrays timestamps must not mutate FeatureMatrix."""
        arrays = _make_arrays(50)
        orig_open_0 = int(arrays.open_times[0])
        orig_close_0 = int(arrays.close_times[0])

        matrix = assemble_feature_matrix(arrays)

        # Mutate the source arrays in-place
        arrays.open_times[0] = 999_999_999
        arrays.close_times[0] = 888_888_888

        # FeatureMatrix must retain original copied values
        assert matrix.open_times[0] == orig_open_0
        assert matrix.close_times[0] == orig_close_0
        assert matrix.get_column("open_times")[0] == orig_open_0
        assert matrix.get_column("close_times")[0] == orig_close_0

    def test_symbol_interval_preserved(self, matrix: FeatureMatrix, large_arrays: NumericalCandleArrays):
        assert matrix.symbol == large_arrays.symbol
        assert matrix.interval == large_arrays.interval

    def test_metadata_fields(self, matrix: FeatureMatrix, large_arrays: NumericalCandleArrays):
        m = matrix.metadata
        assert m.symbol == large_arrays.symbol
        assert m.interval == large_arrays.interval
        assert m.candle_count == large_arrays.length
        assert m.segment_count == 1
        assert m.start_time == int(large_arrays.open_times[0])
        assert m.end_time == int(large_arrays.open_times[-1])
        assert len(m.spec_hash) == 64
        assert m.generated_at_ms > 0

    def test_no_gap_boundaries_single_segment(self, matrix: FeatureMatrix):
        assert not np.any(matrix.metadata.is_gap_boundary)

    def test_warmup_false_at_start(self, matrix: FeatureMatrix):
        """The first bar should never be warmed up."""
        assert not matrix.metadata.is_warmed_up[0]

    def test_warmup_true_after_threshold(self, matrix: FeatureMatrix):
        """After the maximum warmup threshold, is_warmed_up should be True."""
        max_threshold = max(FEATURE_WARMUP_THRESHOLDS.values())
        if len(matrix) > max_threshold:
            assert matrix.metadata.is_warmed_up[max_threshold]

    def test_state_features_empty(self, matrix: FeatureMatrix):
        assert matrix.state_features == {}


# ===========================================================================
# 4. FIRST_VALID_INDEX VERIFICATION
# ===========================================================================


class TestFirstValidIndex:
    """Verify each feature produces NaN before its first_valid_index and finite at it."""

    @pytest.fixture
    def matrix(self) -> FeatureMatrix:
        # 250 bars: enough for most features (not ema200/slope_ema200 which need 200+)
        arrays = _make_arrays(250)
        return assemble_feature_matrix(arrays)

    @pytest.mark.parametrize("name", [
        n for n in CANONICAL_FEATURE_NAMES
        if FEATURE_FIRST_VALID_INDICES[n] < 250
    ])
    def test_first_valid(self, matrix: FeatureMatrix, name: str):
        arr = matrix.features[name]
        fvi = FEATURE_FIRST_VALID_INDICES[name]

        # All values before first_valid_index should be NaN
        if fvi > 0:
            assert np.all(np.isnan(arr[:fvi])), (
                f"{name}: expected all NaN before index {fvi}, "
                f"found finite at {np.where(np.isfinite(arr[:fvi]))[0]}"
            )

        # Value at first_valid_index should be finite
        assert np.isfinite(arr[fvi]), (
            f"{name}: expected finite at index {fvi}, got {arr[fvi]}"
        )


# ===========================================================================
# 5. SEGMENT / GAP ISOLATION
# ===========================================================================


class TestSegmentIsolation:
    """Verify that features are computed independently per segment."""

    def test_gap_boundary_mask(self):
        mask = _build_gap_boundary_mask([10, 20, 5], 35)
        assert mask.shape == (35,)
        assert not mask[0]   # first bar of first segment: not a gap boundary
        assert mask[10]      # first bar of second segment
        assert mask[30]      # first bar of third segment
        # all other False
        assert np.sum(mask) == 2

    def test_warmup_mask_resets_per_segment(self):
        """Each segment starts warm-up counting from zero."""
        max_thresh = max(FEATURE_WARMUP_THRESHOLDS.values())
        # Two segments, one large enough for warm-up, one too small
        mask = _build_warmup_mask([max_thresh + 10, 5], max_thresh + 15)
        # First segment: warmed up from index max_thresh onward
        assert not mask[max_thresh - 1]
        assert mask[max_thresh]
        assert mask[max_thresh + 9]
        # Second segment starts at offset max_thresh + 10
        offset = max_thresh + 10
        for j in range(5):
            assert not mask[offset + j], f"Second segment bar {j} should not be warmed up"

    def test_multi_segment_assembly_via_candles(self):
        """Assemble from candle sequence with an interior gap."""
        candles = _make_candles(100, gap_at=50)
        # With the gap at index 50, this should produce 2 segments
        segments = partition_contiguous_segments(candles, interval="1h", symbol="TEST")
        assert len(segments) == 2

        matrix = assemble_features_from_segments(
            segments, symbol="TEST", interval="1h"
        )
        assert len(matrix) == 100
        assert matrix.metadata.segment_count == 2
        assert matrix.metadata.is_gap_boundary[50]  # gap at index 50
        assert np.sum(matrix.metadata.is_gap_boundary) == 1

    def test_segment_isolation_no_state_leakage(self):
        """Features at the start of segment 2 should be NaN for features
        with nonzero first_valid_index, proving no state carried over."""
        candles = _make_candles(100, gap_at=50)
        segments = partition_contiguous_segments(candles, interval="1h", symbol="TEST")
        matrix = assemble_features_from_segments(
            segments, symbol="TEST", interval="1h"
        )

        # ema20: first_valid = 19. In segment 2 (starting at global index 50),
        # bar 50 should be NaN for ema20 because segment 2 only has 50 bars
        # and ema20 needs 20.
        ema20 = matrix.features["ema20"]
        # Bar 50 is segment 2, bar 0 — should be NaN
        assert np.isnan(ema20[50])


# ===========================================================================
# 6. FUTURE-CANDLE INVARIANCE (POINT-IN-TIME)
# ===========================================================================


class TestFutureCandleInvariance:
    """Adding future candles must not alter any previously computed row."""

    def test_temporal_invariance(self):
        n1 = 220
        n2 = 250
        # Generate the full 250-bar dataset
        arrays_long = _make_arrays(n2)

        # Create the 220-bar prefix by slicing (guarantees identical data)
        arrays_short = NumericalCandleArrays(
            symbol=arrays_long.symbol,
            interval=arrays_long.interval,
            length=n1,
            open_times=arrays_long.open_times[:n1].copy(),
            close_times=arrays_long.close_times[:n1].copy(),
            opens=arrays_long.opens[:n1].copy(),
            highs=arrays_long.highs[:n1].copy(),
            lows=arrays_long.lows[:n1].copy(),
            closes=arrays_long.closes[:n1].copy(),
            volumes=arrays_long.volumes[:n1].copy(),
            quote_volumes=arrays_long.quote_volumes[:n1].copy(),
            trades=arrays_long.trades[:n1].copy(),
            taker_buy_base=arrays_long.taker_buy_base[:n1].copy(),
            taker_buy_quote=arrays_long.taker_buy_quote[:n1].copy(),
        )

        matrix_short = assemble_feature_matrix(arrays_short)
        matrix_long = assemble_feature_matrix(arrays_long)

        for name in CANONICAL_FEATURE_NAMES:
            short_arr = matrix_short.features[name]
            long_arr = matrix_long.features[name][:n1]

            # Use assert_array_equal with equal_nan semantics
            np.testing.assert_array_equal(
                short_arr, long_arr,
                err_msg=f"Future-candle invariance violated for {name}"
            )


# ===========================================================================
# 7. DETERMINISM
# ===========================================================================


class TestDeterminism:
    """Same input must produce identical feature values (ignoring generated_at_ms)."""

    def test_deterministic_features(self):
        arrays = _make_arrays(100)
        m1 = assemble_feature_matrix(arrays)
        m2 = assemble_feature_matrix(arrays)

        for name in CANONICAL_FEATURE_NAMES:
            np.testing.assert_array_equal(
                m1.features[name], m2.features[name],
                err_msg=f"Non-deterministic output for {name}"
            )

    def test_deterministic_spec_hash(self):
        arrays = _make_arrays(100)
        m1 = assemble_feature_matrix(arrays)
        m2 = assemble_feature_matrix(arrays)
        assert m1.metadata.spec_hash == m2.metadata.spec_hash


# ===========================================================================
# 8. INVALID INPUT REJECTION
# ===========================================================================


class TestInvalidInputs:
    """Verify graceful error handling for invalid inputs."""

    def test_non_numerical_arrays_type(self):
        with pytest.raises(FeatureAssemblyError, match="Expected NumericalCandleArrays"):
            assemble_feature_matrix("not_an_arrays")

    def test_empty_segments_list(self):
        with pytest.raises(FeatureAssemblyError, match="empty segment list"):
            assemble_features_from_segments([], symbol="X", interval="1h")


# ===========================================================================
# 9. GET_COLUMN AND COLUMN_NAMES
# ===========================================================================


class TestFeatureMatrixAccessors:
    """Verify FeatureMatrix accessor methods work correctly."""

    @pytest.fixture
    def matrix(self) -> FeatureMatrix:
        return assemble_feature_matrix(_make_arrays(50))

    def test_get_column_feature(self, matrix: FeatureMatrix):
        arr = matrix.get_column("ema20")
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64

    def test_get_column_timestamps(self, matrix: FeatureMatrix):
        ot = matrix.get_column("open_times")
        assert ot.dtype == np.int64

    def test_get_column_missing(self, matrix: FeatureMatrix):
        with pytest.raises(KeyError):
            matrix.get_column("nonexistent_feature")

    def test_column_names_includes_all(self, matrix: FeatureMatrix):
        names = matrix.column_names()
        assert "open_times" in names
        assert "close_times" in names
        for feat_name in CANONICAL_FEATURE_NAMES:
            assert feat_name in names

    def test_shape_property(self, matrix: FeatureMatrix):
        rows, cols = matrix.shape
        assert rows == 50
        # 2 timestamp columns + 30 features + 0 state features
        assert cols == 32


# ===========================================================================
# 10. assemble_features_from_candles END-TO-END
# ===========================================================================


class TestAssembleFromCandles:
    """Test the full candle-to-matrix pipeline."""

    def test_basic_pipeline(self):
        candles = _make_candles(100)
        matrix = assemble_features_from_candles(
            candles, interval="1h", symbol="TESTPAIR"
        )
        assert len(matrix) == 100
        assert matrix.symbol == "TESTPAIR"
        assert matrix.interval == "1h"
        assert len(matrix.features) == 30

    def test_pipeline_with_gap(self):
        candles = _make_candles(100, gap_at=50)
        matrix = assemble_features_from_candles(
            candles, interval="1h", symbol="GAP"
        )
        assert matrix.metadata.segment_count == 2

    def test_pipeline_empty_raises(self):
        with pytest.raises(FeatureAssemblyError):
            assemble_features_from_candles([], interval="1h", symbol="X")


# ===========================================================================
# 11. SMALL SEGMENT (ALL NaN)
# ===========================================================================


class TestSmallSegment:
    """Features should be all NaN when the segment is too short."""

    def test_segment_shorter_than_all_features(self):
        """A 5-bar segment: all features except taker_buy_share should be NaN."""
        arrays = _make_arrays(5)
        matrix = assemble_feature_matrix(arrays)

        for name in CANONICAL_FEATURE_NAMES:
            fvi = FEATURE_FIRST_VALID_INDICES[name]
            if fvi >= 5:
                # All should be NaN
                assert np.all(np.isnan(matrix.features[name])), (
                    f"{name} should be all NaN for 5-bar segment "
                    f"(first_valid={fvi})"
                )

    def test_taker_buy_share_valid_from_zero(self):
        """taker_buy_share has first_valid=0, should have finite values even for 5 bars."""
        arrays = _make_arrays(5)
        matrix = assemble_feature_matrix(arrays)
        assert np.all(np.isfinite(matrix.features["taker_buy_share"]))
