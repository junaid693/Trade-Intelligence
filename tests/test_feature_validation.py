"""Comprehensive tests for Phase 2.1 Unit 2.1 Feature Engine Foundation.

Covers:
- Input validation (chronology, duplicates, spacing, forming candles, timestamps, intervals)
- OHLC consistency and non-finite / non-positive value rejection
- Decimal -> float64 numerical boundary conversion
- Contiguous candle segmentation and gap partitioning
- FeatureSpec identity and canonical SHA-256 hash determinism
- Locked defaults (StructureSpec k=2, EMA slope k=3)
- FeatureMatrix container and metadata verification
"""

from decimal import Decimal
import numpy as np
import pytest

from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.features.exceptions import (
    CorruptedCandleError,
    CorruptedCandleValueError,
    DuplicateTimestampError,
    EmptySequenceError,
    FormingCandleError,
    InvalidIntervalError,
    InvalidTimestampError,
    MalformedIntervalSpacingError,
    UnsortedSequenceError,
)
from trade_intelligence.features.types import (
    CandleSegment,
    FeatureCandle,
    FeatureMatrix,
    FeatureMetadata,
    FeatureSpec,
    MomentumSpec,
    NumericalCandleArrays,
    StructureSpec,
    TrendSpec,
    VolatilitySpec,
    VolumeSpec,
)
from trade_intelligence.features.validation import (
    convert_candles_to_arrays,
    partition_contiguous_segments,
    validate_candle,
    validate_candle_sequence,
)


def _make_candle(
    open_time: int,
    open_price: str = "100.0",
    high_price: str = "105.0",
    low_price: str = "95.0",
    close_price: str = "102.0",
    volume: str = "10.0",
    quote_asset_volume: str = "1000.0",
    number_of_trades: int = 50,
    taker_buy_base: str = "4.0",
    taker_buy_quote: str = "400.0",
    interval_ms: int = 3600000,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
) -> FeatureCandle:
    """Helper to construct a valid FeatureCandle."""
    return FeatureCandle(
        open_time=open_time,
        close_time=open_time + interval_ms - 1,
        open_price=Decimal(open_price),
        high_price=Decimal(high_price),
        low_price=Decimal(low_price),
        close_price=Decimal(close_price),
        volume=Decimal(volume),
        quote_asset_volume=Decimal(quote_asset_volume),
        number_of_trades=number_of_trades,
        taker_buy_base_volume=Decimal(taker_buy_base),
        taker_buy_quote_volume=Decimal(taker_buy_quote),
        symbol=symbol,
        interval=interval,
    )


# ===========================================================================
# 1. Input Validation Tests
# ===========================================================================

class TestSequenceValidation:
    def test_valid_chronological_sequence(self):
        """A properly spaced, chronologically ascending sequence passes validation."""
        candles = [
            _make_candle(open_time=1000000),
            _make_candle(open_time=1000000 + 3600000),
            _make_candle(open_time=1000000 + 2 * 3600000),
        ]
        validated = validate_candle_sequence(candles, interval="1h")
        assert len(validated) == 3
        assert validated[0].open_time == 1000000
        assert validated[2].open_time == 1000000 + 2 * 3600000

    def test_empty_sequence_rejected(self):
        """Empty sequence raises EmptySequenceError."""
        with pytest.raises(EmptySequenceError, match="Candle sequence cannot be empty"):
            validate_candle_sequence([], interval="1h")

    def test_unsorted_sequence_rejected(self):
        """Unsorted timestamps raise UnsortedSequenceError."""
        candles = [
            _make_candle(open_time=2000000),
            _make_candle(open_time=1000000),
        ]
        with pytest.raises(UnsortedSequenceError, match="Unsorted candle sequence"):
            validate_candle_sequence(candles, interval="1h")

    def test_duplicate_timestamps_rejected(self):
        """Duplicate open_time raises DuplicateTimestampError."""
        candles = [
            _make_candle(open_time=1000000),
            _make_candle(open_time=1000000),
        ]
        with pytest.raises(DuplicateTimestampError, match="Duplicate open_time"):
            validate_candle_sequence(candles, interval="1h")

    def test_malformed_interval_spacing_shorter_than_interval(self):
        """Delta < expected_interval_ms raises MalformedIntervalSpacingError."""
        candles = [
            _make_candle(open_time=1000000),
            _make_candle(open_time=1000000 + 1800000),  # Only 30m delta on 1h interval
        ]
        with pytest.raises(MalformedIntervalSpacingError, match="less than expected interval duration"):
            validate_candle_sequence(candles, interval="1h")

    def test_interior_gap_allowed_in_sequence_validation(self):
        """An interior gap (delta > expected_interval_ms) is accepted by validate_candle_sequence."""
        candles = [
            _make_candle(open_time=1000000),
            _make_candle(open_time=1000000 + 2 * 3600000),  # Missing 1h bar in between
        ]
        validated = validate_candle_sequence(candles, interval="1h")
        assert len(validated) == 2

    def test_invalid_interval_rejected(self):
        """Invalid interval string raises InvalidIntervalError."""
        candles = [_make_candle(open_time=1000000)]
        with pytest.raises(InvalidIntervalError, match="Invalid KlineInterval"):
            validate_candle_sequence(candles, interval="2h")

    def test_invalid_timestamp_rejected(self):
        """Negative or boolean timestamps raise InvalidTimestampError."""
        with pytest.raises(InvalidTimestampError):
            validate_candle({"open_time": -1, "close_time": 100, "open_price": "1", "high_price": "1", "low_price": "1", "close_price": "1"})

        with pytest.raises(InvalidTimestampError):
            validate_candle({"open_time": True, "close_time": 100, "open_price": "1", "high_price": "1", "low_price": "1", "close_price": "1"})

    def test_forming_candle_rejected(self):
        """A candle whose close_time >= server_time_ms raises FormingCandleError."""
        candle = _make_candle(open_time=1000000, interval_ms=3600000)
        # close_time is 1000000 + 3600000 - 1 = 4599999
        server_time = 4599999  # exact close_time boundary
        with pytest.raises(FormingCandleError, match="Candle is forming"):
            validate_candle(candle, server_time_ms=server_time)

        # But when candle has closed prior to server_time, it passes
        valid = validate_candle(candle, server_time_ms=4600000)
        assert valid.open_time == 1000000


# ===========================================================================
# 2. OHLC Physical Consistency & Value Validation Tests
# ===========================================================================

class TestCandleValueValidation:
    def test_impossible_high_low_relationships(self):
        """High < Low, High < Open, or Low > Close raises CorruptedCandleError."""
        # High < Low
        with pytest.raises(CorruptedCandleError, match="cannot be less than low_price"):
            validate_candle({
                "open_time": 1000, "close_time": 2000,
                "open_price": "100", "high_price": "90", "low_price": "95", "close_price": "92",
            })

        # High < Open
        with pytest.raises(CorruptedCandleError, match="must be >= open_price"):
            validate_candle({
                "open_time": 1000, "close_time": 2000,
                "open_price": "105", "high_price": "100", "low_price": "90", "close_price": "95",
            })

        # Low > Close
        with pytest.raises(CorruptedCandleError, match="must be <= open_price"):
            validate_candle({
                "open_time": 1000, "close_time": 2000,
                "open_price": "100", "high_price": "105", "low_price": "98", "close_price": "95",
            })

    def test_non_positive_prices_rejected(self):
        """Zero or negative prices raise CorruptedCandleValueError."""
        with pytest.raises(CorruptedCandleValueError, match="must be > 0"):
            validate_candle({
                "open_time": 1000, "close_time": 2000,
                "open_price": "0", "high_price": "10", "low_price": "0", "close_price": "5",
            })

        with pytest.raises(CorruptedCandleValueError, match="must be > 0"):
            validate_candle({
                "open_time": 1000, "close_time": 2000,
                "open_price": "-10", "high_price": "10", "low_price": "-10", "close_price": "5",
            })

    def test_non_finite_values_rejected(self):
        """NaN and Infinity values raise CorruptedCandleValueError."""
        with pytest.raises(CorruptedCandleValueError, match="must be finite|finite Decimal"):
            validate_candle({
                "open_time": 1000, "close_time": 2000,
                "open_price": "NaN", "high_price": "10", "low_price": "1", "close_price": "5",
            })

        with pytest.raises(CorruptedCandleValueError, match="must be finite|finite Decimal"):
            validate_candle({
                "open_time": 1000, "close_time": 2000,
                "open_price": "10", "high_price": "Infinity", "low_price": "1", "close_price": "5",
            })

    def test_invalid_volume_rejected(self):
        """Negative volume or taker volume > total volume raises CorruptedCandleValueError."""
        # Negative volume
        with pytest.raises(CorruptedCandleValueError, match="cannot be negative"):
            validate_candle({
                "open_time": 1000, "close_time": 2000,
                "open_price": "10", "high_price": "12", "low_price": "9", "close_price": "11",
                "volume": "-1",
            })

        # Taker volume exceeding total volume
        with pytest.raises(CorruptedCandleValueError, match="cannot exceed total volume"):
            validate_candle({
                "open_time": 1000, "close_time": 2000,
                "open_price": "10", "high_price": "12", "low_price": "9", "close_price": "11",
                "volume": "10", "taker_buy_base_volume": "11",
            })


# ===========================================================================
# 3. Decimal -> Float64 Conversion Boundary Tests
# ===========================================================================

class TestNumericalConversionBoundary:
    def test_valid_conversion_to_float64_arrays(self):
        """Valid candles convert into 1D np.float64 and np.int64 contiguous arrays."""
        candles = [
            _make_candle(open_time=1000000, open_price="100.5", close_price="102.25", volume="50.123"),
            _make_candle(open_time=1000000 + 3600000, open_price="102.25", close_price="104.75", volume="60.456"),
        ]
        arrays = convert_candles_to_arrays(candles, symbol="BTCUSDT", interval="1h")
        assert isinstance(arrays, NumericalCandleArrays)
        assert arrays.length == 2
        assert arrays.opens.dtype == np.float64
        assert arrays.closes.dtype == np.float64
        assert arrays.highs.dtype == np.float64
        assert arrays.lows.dtype == np.float64
        assert arrays.volumes.dtype == np.float64
        assert arrays.quote_volumes.dtype == np.float64
        assert arrays.taker_buy_base.dtype == np.float64
        assert arrays.taker_buy_quote.dtype == np.float64

        assert arrays.open_times.dtype == np.int64
        assert arrays.close_times.dtype == np.int64
        assert arrays.trades.dtype == np.int64

        assert arrays.opens.ndim == 1
        assert arrays.closes.ndim == 1
        assert arrays.opens[0] == 100.5
        assert arrays.closes[1] == 104.75

    def test_empty_sequence_conversion_raises(self):
        """Converting an empty sequence raises EmptySequenceError."""
        with pytest.raises(EmptySequenceError, match="Cannot convert empty candle sequence"):
            convert_candles_to_arrays([], symbol="BTCUSDT", interval="1h")

    def test_decimal_nan_rejected_at_conversion_boundary(self):
        """Decimal('NaN') is rejected before conversion with CorruptedCandleValueError."""
        # Price field as NaN
        nan_price_candle = FeatureCandle(
            open_time=1000,
            close_time=2000,
            open_price=Decimal("100.0"),
            high_price=Decimal("105.0"),
            low_price=Decimal("95.0"),
            close_price=Decimal("NaN"),
            volume=Decimal("10.0"),
            quote_asset_volume=Decimal("1000.0"),
            number_of_trades=10,
            taker_buy_base_volume=Decimal("5.0"),
            taker_buy_quote_volume=Decimal("500.0"),
        )
        with pytest.raises(CorruptedCandleValueError, match="non-finite value .* rejected before conversion|must be a finite Decimal"):
            convert_candles_to_arrays([nan_price_candle], symbol="BTCUSDT", interval="1h")

        # Volume field as NaN
        nan_vol_candle = FeatureCandle(
            open_time=1000,
            close_time=2000,
            open_price=Decimal("100.0"),
            high_price=Decimal("105.0"),
            low_price=Decimal("95.0"),
            close_price=Decimal("100.0"),
            volume=Decimal("NaN"),
            quote_asset_volume=Decimal("1000.0"),
            number_of_trades=10,
            taker_buy_base_volume=Decimal("5.0"),
            taker_buy_quote_volume=Decimal("500.0"),
        )
        with pytest.raises(CorruptedCandleValueError, match="non-finite value .* rejected before conversion|must be a finite Decimal"):
            convert_candles_to_arrays([nan_vol_candle], symbol="BTCUSDT", interval="1h")

    def test_decimal_infinity_rejected_at_conversion_boundary(self):
        """Decimal('Infinity') is rejected before conversion with CorruptedCandleValueError."""
        inf_candle = FeatureCandle(
            open_time=1000,
            close_time=2000,
            open_price=Decimal("100.0"),
            high_price=Decimal("Infinity"),
            low_price=Decimal("95.0"),
            close_price=Decimal("100.0"),
            volume=Decimal("10.0"),
            quote_asset_volume=Decimal("1000.0"),
            number_of_trades=10,
            taker_buy_base_volume=Decimal("5.0"),
            taker_buy_quote_volume=Decimal("500.0"),
        )
        with pytest.raises(CorruptedCandleValueError, match="non-finite value .* rejected before conversion|must be a finite Decimal"):
            convert_candles_to_arrays([inf_candle], symbol="BTCUSDT", interval="1h")

    def test_decimal_negative_infinity_rejected_at_conversion_boundary(self):
        """Decimal('-Infinity') is rejected before conversion with CorruptedCandleValueError."""
        neg_inf_candle = FeatureCandle(
            open_time=1000,
            close_time=2000,
            open_price=Decimal("100.0"),
            high_price=Decimal("105.0"),
            low_price=Decimal("-Infinity"),
            close_price=Decimal("100.0"),
            volume=Decimal("10.0"),
            quote_asset_volume=Decimal("1000.0"),
            number_of_trades=10,
            taker_buy_base_volume=Decimal("5.0"),
            taker_buy_quote_volume=Decimal("500.0"),
        )
        with pytest.raises(CorruptedCandleValueError, match="non-finite value .* rejected before conversion|must be a finite Decimal"):
            convert_candles_to_arrays([neg_inf_candle], symbol="BTCUSDT", interval="1h")

    def test_invalid_decimal_values_rejected_before_conversion(self):
        """Invalid Decimal values (non-Decimal types, non-positive price, negative volume) are rejected before conversion."""
        # Non-Decimal price
        bad_type_candle = FeatureCandle(
            open_time=1000,
            close_time=2000,
            open_price=100.5,  # type: ignore[arg-type]
            high_price=Decimal("105.0"),
            low_price=Decimal("95.0"),
            close_price=Decimal("100.0"),
            volume=Decimal("10.0"),
            quote_asset_volume=Decimal("1000.0"),
            number_of_trades=10,
            taker_buy_base_volume=Decimal("5.0"),
            taker_buy_quote_volume=Decimal("500.0"),
        )
        with pytest.raises(CorruptedCandleValueError, match="expected Decimal"):
            convert_candles_to_arrays([bad_type_candle], symbol="BTCUSDT", interval="1h")

        # Non-positive price
        zero_price_candle = FeatureCandle(
            open_time=1000,
            close_time=2000,
            open_price=Decimal("0"),
            high_price=Decimal("105.0"),
            low_price=Decimal("95.0"),
            close_price=Decimal("100.0"),
            volume=Decimal("10.0"),
            quote_asset_volume=Decimal("1000.0"),
            number_of_trades=10,
            taker_buy_base_volume=Decimal("5.0"),
            taker_buy_quote_volume=Decimal("500.0"),
        )
        with pytest.raises(CorruptedCandleValueError, match="must be positive"):
            convert_candles_to_arrays([zero_price_candle], symbol="BTCUSDT", interval="1h")

        # Negative volume
        neg_vol_candle = FeatureCandle(
            open_time=1000,
            close_time=2000,
            open_price=Decimal("100.0"),
            high_price=Decimal("105.0"),
            low_price=Decimal("95.0"),
            close_price=Decimal("100.0"),
            volume=Decimal("-5.0"),
            quote_asset_volume=Decimal("1000.0"),
            number_of_trades=10,
            taker_buy_base_volume=Decimal("5.0"),
            taker_buy_quote_volume=Decimal("500.0"),
        )
        with pytest.raises(CorruptedCandleValueError, match="cannot be negative"):
            convert_candles_to_arrays([neg_vol_candle], symbol="BTCUSDT", interval="1h")


# ===========================================================================
# 4. Contiguous Segment Partitioning Tests
# ===========================================================================

class TestContiguousSegmentation:
    def test_completely_continuous_sequence_produces_one_segment(self):
        """A contiguous sequence with zero gaps produces exactly one segment."""
        candles = [
            _make_candle(open_time=1000000 + i * 3600000) for i in range(10)
        ]
        segments = partition_contiguous_segments(candles, interval="1h", symbol="BTCUSDT")
        assert len(segments) == 1
        assert segments[0].length == 10
        assert segments[0].start_open_time == 1000000
        assert segments[0].end_open_time == 1000000 + 9 * 3600000
        assert len(segments[0].candles) == 10

    def test_one_interior_gap_produces_two_segments(self):
        """An interior gap partitions the sequence into two distinct segments."""
        # 5 candles, gap of 2h, then 5 candles
        seg1 = [_make_candle(open_time=1000000 + i * 3600000) for i in range(5)]
        gap_start = 1000000 + 4 * 3600000 + 2 * 3600000  # 1 missing bar
        seg2 = [_make_candle(open_time=gap_start + i * 3600000) for i in range(5)]
        all_candles = seg1 + seg2

        segments = partition_contiguous_segments(all_candles, interval="1h", symbol="BTCUSDT")
        assert len(segments) == 2
        assert segments[0].length == 5
        assert segments[0].start_open_time == 1000000
        assert segments[0].end_open_time == 1000000 + 4 * 3600000

        assert segments[1].length == 5
        assert segments[1].start_open_time == gap_start
        assert segments[1].end_open_time == gap_start + 4 * 3600000

    def test_multiple_gaps_produce_multiple_segments(self):
        """Three separate clusters produce three distinct segments."""
        t0 = 1000000
        c1 = [_make_candle(open_time=t0 + i * 3600000) for i in range(3)]
        t1 = t0 + 5 * 3600000  # gap of 2 missing bars
        c2 = [_make_candle(open_time=t1 + i * 3600000) for i in range(4)]
        t2 = t1 + 8 * 3600000  # another gap
        c3 = [_make_candle(open_time=t2 + i * 3600000) for i in range(2)]

        all_candles = c1 + c2 + c3
        segments = partition_contiguous_segments(all_candles, interval="1h", symbol="BTCUSDT")
        assert len(segments) == 3
        assert segments[0].length == 3
        assert segments[1].length == 4
        assert segments[2].length == 2

    def test_segmentation_preserves_original_candle_data(self):
        """No synthetic candles are created and original candle records are identical."""
        candles = [
            _make_candle(open_time=1000000, close_price="101.5", high_price="105.0"),
            _make_candle(open_time=1000000 + 2 * 3600000, close_price="104.5", high_price="105.0"),
        ]
        segments = partition_contiguous_segments(candles, interval="1h", symbol="BTCUSDT")
        assert len(segments) == 2
        assert segments[0].candles[0].close_price == Decimal("101.5")
        assert segments[1].candles[0].close_price == Decimal("104.5")


# ===========================================================================
# 5. FeatureSpec Identity & Hash Determinism Tests
# ===========================================================================

class TestFeatureSpecIdentity:
    def test_identical_specs_produce_identical_hashes(self):
        """Identical and equivalent configurations produce identical full 64-character SHA-256 spec_hash."""
        spec1 = FeatureSpec(name="ema", version="1.0.0", params={"period": 20, "source": "close"})
        spec2 = FeatureSpec(name="ema", version="1.0.0", params={"period": 20, "source": "close"})

        # Deterministic hash: calling spec_hash repeatedly yields identical result
        assert spec1.spec_hash == spec1.spec_hash

        # Equivalent specs produce identical full hashes
        assert spec1.spec_hash == spec2.spec_hash

        # Hash length is exactly 64 hexadecimal characters
        assert len(spec1.spec_hash) == 64
        assert all(c in "0123456789abcdef" for c in spec1.spec_hash)
        assert int(spec1.spec_hash, 16) >= 0

    def test_canonical_serialization_independent_of_dict_order(self):
        """Key insertion order in params dict does not affect full 64-character spec_hash."""
        spec1 = FeatureSpec(name="rsi", params={"period": 14, "smoothing": "wilder", "source": "close"})
        spec2 = FeatureSpec(name="rsi", params={"source": "close", "period": 14, "smoothing": "wilder"})
        assert spec1.spec_hash == spec2.spec_hash
        assert len(spec1.spec_hash) == 64
        assert all(c in "0123456789abcdef" for c in spec1.spec_hash)

    def test_changing_period_changes_hash(self):
        """Altering indicator period alters spec_hash."""
        spec1 = FeatureSpec(name="ema", params={"period": 20, "source": "close"})
        spec2 = FeatureSpec(name="ema", params={"period": 50, "source": "close"})
        assert spec1.spec_hash != spec2.spec_hash

    def test_changing_price_source_changes_hash(self):
        """Altering price source alters spec_hash."""
        spec1 = FeatureSpec(name="ema", params={"period": 20, "source": "close"})
        spec2 = FeatureSpec(name="ema", params={"period": 20, "source": "open"})
        assert spec1.spec_hash != spec2.spec_hash

    def test_changing_smoothing_changes_hash(self):
        """Altering smoothing algorithm alters spec_hash."""
        spec1 = FeatureSpec(name="ma", params={"period": 20, "smoothing": "exponential"})
        spec2 = FeatureSpec(name="ma", params={"period": 20, "smoothing": "simple"})
        assert spec1.spec_hash != spec2.spec_hash

    def test_changing_seed_convention_changes_hash(self):
        """Altering moving average seed convention alters spec_hash."""
        spec1 = FeatureSpec(name="ema", params={"period": 20, "seed_convention": "sma"})
        spec2 = FeatureSpec(name="ema", params={"period": 20, "seed_convention": "first_value"})
        assert spec1.spec_hash != spec2.spec_hash

    def test_changing_ddof_changes_hash(self):
        """Altering Bollinger standard deviation ddof alters spec_hash."""
        spec1 = FeatureSpec(name="bollinger", params={"period": 20, "ddof": 0})
        spec2 = FeatureSpec(name="bollinger", params={"period": 20, "ddof": 1})
        assert spec1.spec_hash != spec2.spec_hash

    def test_changing_slope_k_changes_hash(self):
        """Altering EMA slope lookback k alters spec_hash."""
        spec1 = TrendSpec(ema_periods=(20,), slope_k=3).to_feature_spec()
        spec2 = TrendSpec(ema_periods=(20,), slope_k=5).to_feature_spec()
        assert spec1.spec_hash != spec2.spec_hash

    def test_changing_pivot_k_changes_hash(self):
        """Altering structure pivot window k alters spec_hash."""
        spec1 = StructureSpec(pivot_k=2).to_feature_spec()
        spec2 = StructureSpec(pivot_k=3).to_feature_spec()
        assert spec1.spec_hash != spec2.spec_hash

    def test_changing_operational_constant_changes_hash(self):
        """Altering an operational constant (e.g. warmup_multiplier) alters spec_hash."""
        spec1 = TrendSpec(warmup_multiplier=3).to_feature_spec()
        spec2 = TrendSpec(warmup_multiplier=4).to_feature_spec()
        assert spec1.spec_hash != spec2.spec_hash


# ===========================================================================
# 6. Locked Defaults Tests
# ===========================================================================

class TestLockedDefaults:
    def test_structure_spec_locked_defaults(self):
        """StructureSpec defaults: pivot_k=2 (5-bar window), strict_inequality=True."""
        spec = StructureSpec()
        assert spec.pivot_k == 2
        assert spec.strict_inequality is True
        assert spec.range_periods == (20, 50)
        f_spec = spec.to_feature_spec()
        assert f_spec.params["pivot_k"] == 2
        assert f_spec.params["window_size"] == 5
        assert f_spec.params["strict_inequality"] is True

    def test_trend_spec_locked_defaults(self):
        """TrendSpec defaults: slope_k=3, ema_periods=(20, 50, 200), warmup_multiplier=3."""
        spec = TrendSpec()
        assert spec.slope_k == 3
        assert spec.ema_periods == (20, 50, 200)
        assert spec.warmup_multiplier == 3

    def test_volatility_spec_locked_defaults(self):
        """VolatilitySpec defaults: ddof=0 (population standard deviation)."""
        spec = VolatilitySpec()
        assert spec.ddof == 0
        assert spec.bb_std_mult == 2.0
        assert spec.atr_period == 14

    def test_volume_spec_locked_defaults(self):
        """VolumeSpec defaults: neutral_taker_share=0.5, period=20."""
        spec = VolumeSpec()
        assert spec.neutral_taker_share == 0.5
        assert spec.volume_sma_period == 20


# ===========================================================================
# 7. FeatureMatrix Container & Metadata Tests
# ===========================================================================

class TestFeatureMatrixContainer:
    def test_feature_matrix_valid_construction(self):
        """FeatureMatrix correctly stores named 1D float64 arrays and metadata."""
        count = 5
        open_times = np.arange(count, dtype=np.int64) * 3600000
        close_times = open_times + 3599999
        ema20 = np.linspace(100.0, 105.0, count, dtype=np.float64)
        rsi14 = np.linspace(45.0, 55.0, count, dtype=np.float64)
        swing_tags = np.zeros(count, dtype=np.int8)

        metadata = FeatureMetadata(
            symbol="BTCUSDT",
            interval="1h",
            start_time=int(open_times[0]),
            end_time=int(open_times[-1]),
            candle_count=count,
            segment_count=1,
            is_warmed_up=np.ones(count, dtype=bool),
            is_gap_boundary=np.zeros(count, dtype=bool),
            spec_version="1.0.0",
            spec_hash="a1b2c3d4e5f60718" * 4,
            generated_at_ms=1700000000000,
        )

        matrix = FeatureMatrix(
            symbol="BTCUSDT",
            interval="1h",
            open_times=open_times,
            close_times=close_times,
            features={"ema20": ema20, "rsi14": rsi14},
            state_features={"swing_tags": swing_tags},
            metadata=metadata,
        )

        assert len(matrix) == 5
        assert matrix.shape == (5, 5)  # open_times, close_times, ema20, rsi14, swing_tags
        assert "ema20" in matrix.column_names()
        assert "swing_tags" in matrix.column_names()
        np.testing.assert_array_equal(matrix.get_column("ema20"), ema20)
        np.testing.assert_array_equal(matrix.get_column("open_times"), open_times)

    def test_feature_matrix_rejects_non_1d_array(self):
        """FeatureMatrix rejects non-1D numerical arrays."""
        count = 5
        open_times = np.arange(count, dtype=np.int64)
        close_times = open_times + 1
        bad_2d = np.ones((count, 2), dtype=np.float64)

        metadata = FeatureMetadata(
            symbol="BTCUSDT", interval="1h", start_time=0, end_time=4,
            candle_count=count, segment_count=1,
            is_warmed_up=np.ones(count, dtype=bool),
            is_gap_boundary=np.zeros(count, dtype=bool),
            spec_version="1.0.0", spec_hash="0" * 64, generated_at_ms=1000,
        )

        with pytest.raises(ValueError, match="must be 1D"):
            FeatureMatrix(
                symbol="BTCUSDT", interval="1h",
                open_times=open_times, close_times=close_times,
                features={"bad": bad_2d}, state_features={},
                metadata=metadata,
            )

    def test_feature_matrix_rejects_non_float64_numerical_feature(self):
        """FeatureMatrix rejects numerical features with dtype other than float64."""
        count = 5
        open_times = np.arange(count, dtype=np.int64)
        close_times = open_times + 1
        float32_arr = np.ones(count, dtype=np.float32)

        metadata = FeatureMetadata(
            symbol="BTCUSDT", interval="1h", start_time=0, end_time=4,
            candle_count=count, segment_count=1,
            is_warmed_up=np.ones(count, dtype=bool),
            is_gap_boundary=np.zeros(count, dtype=bool),
            spec_version="1.0.0", spec_hash="0" * 64, generated_at_ms=1000,
        )

        with pytest.raises(TypeError, match="dtype must be float64"):
            FeatureMatrix(
                symbol="BTCUSDT", interval="1h",
                open_times=open_times, close_times=close_times,
                features={"f32": float32_arr}, state_features={},
                metadata=metadata,
            )
