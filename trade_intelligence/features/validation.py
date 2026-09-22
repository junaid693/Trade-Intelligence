"""Input validation and numerical boundary conversion for the feature engine."""

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

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
    NumericalCandleArrays,
)
from trade_intelligence.klines.gap_detector import INTERVAL_MS, interval_to_milliseconds


def _to_decimal(val: Any, name: str) -> Decimal:
    """Validate and convert a price or volume value to a finite Decimal."""
    if isinstance(val, bool):
        raise CorruptedCandleValueError(f"{name} cannot be a boolean: {val!r}")
    if isinstance(val, Decimal):
        d = val
    elif isinstance(val, (int, str)):
        try:
            d = Decimal(str(val))
        except InvalidOperation:
            raise CorruptedCandleValueError(f"Invalid {name} representation: {val!r}")
    elif isinstance(val, float):
        if not np.isfinite(val):
            raise CorruptedCandleValueError(f"{name} must be finite, got: {val}")
        d = Decimal(str(val))
    else:
        raise CorruptedCandleValueError(f"{name} has invalid type {type(val).__name__}: {val!r}")

    if not d.is_finite():
        raise CorruptedCandleValueError(f"{name} must be a finite Decimal, got: {d}")
    return d


def validate_candle(
    candle: Union[FeatureCandle, Dict[str, Any], Any],
    server_time_ms: Optional[int] = None,
) -> FeatureCandle:
    """Validate an individual candlestick and return a normalized FeatureCandle.

    Enforces:
    - Positive finite prices
    - High >= max(Open, Close, Low) and Low <= min(Open, Close, High)
    - Non-negative finite volumes
    - Taker volume <= Total volume
    - Non-negative integer timestamps with close_time > open_time
    - Closed candle rule (close_time < server_time_ms) when server_time_ms is provided
    """
    if isinstance(candle, dict):
        raw_ot = candle.get("open_time")
        raw_ct = candle.get("close_time")
        raw_op = candle.get("open_price")
        raw_hp = candle.get("high_price")
        raw_lp = candle.get("low_price")
        raw_cp = candle.get("close_price")
        raw_vol = candle.get("volume", "0")
        raw_qav = candle.get("quote_asset_volume", "0")
        raw_trades = candle.get("number_of_trades", 0)
        raw_tb = candle.get("taker_buy_base_volume", "0")
        raw_tq = candle.get("taker_buy_quote_volume", "0")
        symbol = candle.get("symbol")
        interval = candle.get("interval")
    elif isinstance(candle, FeatureCandle):
        raw_ot = candle.open_time
        raw_ct = candle.close_time
        raw_op = candle.open_price
        raw_hp = candle.high_price
        raw_lp = candle.low_price
        raw_cp = candle.close_price
        raw_vol = candle.volume
        raw_qav = candle.quote_asset_volume
        raw_trades = candle.number_of_trades
        raw_tb = candle.taker_buy_base_volume
        raw_tq = candle.taker_buy_quote_volume
        symbol = candle.symbol
        interval = candle.interval
    else:
        # Generic object attribute access (e.g. Kline model)
        try:
            raw_ot = getattr(candle, "open_time")
            raw_ct = getattr(candle, "close_time")
            raw_op = getattr(candle, "open_price")
            raw_hp = getattr(candle, "high_price")
            raw_lp = getattr(candle, "low_price")
            raw_cp = getattr(candle, "close_price")
            raw_vol = getattr(candle, "volume", "0")
            raw_qav = getattr(candle, "quote_asset_volume", "0")
            raw_trades = getattr(candle, "number_of_trades", 0)
            raw_tb = getattr(candle, "taker_buy_base_volume", "0")
            raw_tq = getattr(candle, "taker_buy_quote_volume", "0")
            symbol = getattr(candle, "symbol", None)
            interval = getattr(candle, "interval", None)
        except AttributeError as e:
            raise CorruptedCandleError(f"Malformed candle object missing required attributes: {e}")

    # Timestamps
    if isinstance(raw_ot, bool) or not isinstance(raw_ot, int) or raw_ot < 0:
        raise InvalidTimestampError(f"open_time must be non-negative int, got {raw_ot!r}")
    if isinstance(raw_ct, bool) or not isinstance(raw_ct, int) or raw_ct <= raw_ot:
        raise InvalidTimestampError(f"close_time must be int > open_time ({raw_ot}), got {raw_ct!r}")

    # Closed-candle enforcement
    if server_time_ms is not None:
        if isinstance(server_time_ms, bool) or not isinstance(server_time_ms, int) or server_time_ms < 0:
            raise InvalidTimestampError(f"server_time_ms must be non-negative int, got {server_time_ms!r}")
        if raw_ct >= server_time_ms:
            raise FormingCandleError(
                f"Candle is forming: close_time ({raw_ct}) >= server_time ({server_time_ms})"
            )

    # Prices
    op = _to_decimal(raw_op, "open_price")
    hp = _to_decimal(raw_hp, "high_price")
    lp = _to_decimal(raw_lp, "low_price")
    cp = _to_decimal(raw_cp, "close_price")

    if op <= 0:
        raise CorruptedCandleValueError(f"open_price must be > 0, got {op}")
    if hp <= 0:
        raise CorruptedCandleValueError(f"high_price must be > 0, got {hp}")
    if lp <= 0:
        raise CorruptedCandleValueError(f"low_price must be > 0, got {lp}")
    if cp <= 0:
        raise CorruptedCandleValueError(f"close_price must be > 0, got {cp}")

    # OHLC physical relationships
    if hp < lp:
        raise CorruptedCandleError(f"high_price ({hp}) cannot be less than low_price ({lp})")
    if hp < op or hp < cp:
        raise CorruptedCandleError(
            f"high_price ({hp}) must be >= open_price ({op}) and close_price ({cp})"
        )
    if lp > op or lp > cp:
        raise CorruptedCandleError(
            f"low_price ({lp}) must be <= open_price ({op}) and close_price ({cp})"
        )

    # Volumes
    vol = _to_decimal(raw_vol, "volume")
    qav = _to_decimal(raw_qav, "quote_asset_volume")
    tb = _to_decimal(raw_tb, "taker_buy_base_volume")
    tq = _to_decimal(raw_tq, "taker_buy_quote_volume")

    if vol < 0:
        raise CorruptedCandleValueError(f"volume cannot be negative, got {vol}")
    if qav < 0:
        raise CorruptedCandleValueError(f"quote_asset_volume cannot be negative, got {qav}")
    if tb < 0:
        raise CorruptedCandleValueError(f"taker_buy_base_volume cannot be negative, got {tb}")
    if tq < 0:
        raise CorruptedCandleValueError(f"taker_buy_quote_volume cannot be negative, got {tq}")

    if tb > vol:
        raise CorruptedCandleValueError(
            f"taker_buy_base_volume ({tb}) cannot exceed total volume ({vol})"
        )
    if tq > qav:
        raise CorruptedCandleValueError(
            f"taker_buy_quote_volume ({tq}) cannot exceed total quote_asset_volume ({qav})"
        )

    if isinstance(raw_trades, bool) or not isinstance(raw_trades, int) or raw_trades < 0:
        raise CorruptedCandleValueError(f"number_of_trades must be non-negative int, got {raw_trades!r}")

    return FeatureCandle(
        open_time=raw_ot,
        close_time=raw_ct,
        open_price=op,
        high_price=hp,
        low_price=lp,
        close_price=cp,
        volume=vol,
        quote_asset_volume=qav,
        number_of_trades=raw_trades,
        taker_buy_base_volume=tb,
        taker_buy_quote_volume=tq,
        symbol=symbol,
        interval=interval,
    )


def validate_candle_sequence(
    candles: Sequence[Union[FeatureCandle, Dict[str, Any], Any]],
    interval: Union[str, KlineInterval],
    server_time_ms: Optional[int] = None,
) -> List[FeatureCandle]:
    """Validate a sequence of candlesticks strictly.

    Enforces:
    - Non-empty sequence
    - Valid interval
    - Strict chronological monotonicity (open_time[i+1] > open_time[i])
    - No duplicate timestamps
    - Expected interval spacing: delta >= interval_ms (delta < interval_ms raises MalformedIntervalSpacingError)
    - Closed candle rule for all candles when server_time_ms is provided
    """
    if not candles:
        raise EmptySequenceError("Candle sequence cannot be empty.")

    try:
        iv = KlineInterval.from_value(interval)
    except (ValueError, TypeError) as e:
        raise InvalidIntervalError(f"Invalid KlineInterval: {interval!r}") from e

    expected_ms = interval_to_milliseconds(iv.value)

    validated: List[FeatureCandle] = []
    prev_ot: Optional[int] = None

    for i, raw_candle in enumerate(candles):
        c = validate_candle(raw_candle, server_time_ms=server_time_ms)
        ot = c.open_time

        if prev_ot is not None:
            if ot == prev_ot:
                raise DuplicateTimestampError(
                    f"Duplicate open_time ({ot}) detected at sequence index {i}"
                )
            if ot < prev_ot:
                raise UnsortedSequenceError(
                    f"Unsorted candle sequence at index {i}: open_time {ot} < previous {prev_ot}"
                )
            delta = ot - prev_ot
            if delta < expected_ms:
                raise MalformedIntervalSpacingError(
                    f"Malformed interval spacing at index {i}: delta {delta} ms is less than "
                    f"expected interval duration {expected_ms} ms for {iv.value}"
                )

        prev_ot = ot
        validated.append(c)

    return validated


def convert_candles_to_arrays(
    candles: Sequence[FeatureCandle],
    symbol: str = "UNKNOWN",
    interval: str = "1h",
) -> NumericalCandleArrays:
    """Explicit conversion boundary from validated FeatureCandle sequence to NumericalCandleArrays.

    Converts exact Decimal prices and volumes into homogeneous, contiguous
    1D numpy.float64 arrays, and timestamps into 1D numpy.int64 arrays.
    Validates Decimal finiteness and validity before float conversion.
    """
    if not candles:
        raise EmptySequenceError("Cannot convert empty candle sequence to numerical arrays.")

    # Validate all Decimal values before conversion
    for idx, c in enumerate(candles):
        for field_name, is_price in [
            ("open_price", True),
            ("high_price", True),
            ("low_price", True),
            ("close_price", True),
            ("volume", False),
            ("quote_asset_volume", False),
            ("taker_buy_base_volume", False),
            ("taker_buy_quote_volume", False),
        ]:
            val = getattr(c, field_name, None)
            if not isinstance(val, Decimal):
                raise CorruptedCandleValueError(
                    f"Invalid Decimal at index {idx} for {field_name}: expected Decimal, got {type(val).__name__}"
                )
            if not val.is_finite():
                raise CorruptedCandleValueError(
                    f"Invalid Decimal at index {idx} for {field_name}: non-finite value {val} rejected before conversion"
                )
            if is_price and val <= 0:
                raise CorruptedCandleValueError(
                    f"Invalid price at index {idx} for {field_name}: {val} must be positive"
                )
            if not is_price and val < 0:
                raise CorruptedCandleValueError(
                    f"Invalid volume at index {idx} for {field_name}: {val} cannot be negative"
                )

    count = len(candles)
    open_times = np.empty(count, dtype=np.int64)
    close_times = np.empty(count, dtype=np.int64)
    opens = np.empty(count, dtype=np.float64)
    highs = np.empty(count, dtype=np.float64)
    lows = np.empty(count, dtype=np.float64)
    closes = np.empty(count, dtype=np.float64)
    volumes = np.empty(count, dtype=np.float64)
    quote_volumes = np.empty(count, dtype=np.float64)
    trades = np.empty(count, dtype=np.int64)
    taker_buy_base = np.empty(count, dtype=np.float64)
    taker_buy_quote = np.empty(count, dtype=np.float64)

    for i, c in enumerate(candles):
        open_times[i] = c.open_time
        close_times[i] = c.close_time
        opens[i] = float(c.open_price)
        highs[i] = float(c.high_price)
        lows[i] = float(c.low_price)
        closes[i] = float(c.close_price)
        volumes[i] = float(c.volume)
        quote_volumes[i] = float(c.quote_asset_volume)
        trades[i] = c.number_of_trades
        taker_buy_base[i] = float(c.taker_buy_base_volume)
        taker_buy_quote[i] = float(c.taker_buy_quote_volume)

    # Post-conversion sanity assertions: all values must be finite
    for name, arr in [
        ("opens", opens),
        ("highs", highs),
        ("lows", lows),
        ("closes", closes),
        ("volumes", volumes),
        ("quote_volumes", quote_volumes),
        ("taker_buy_base", taker_buy_base),
        ("taker_buy_quote", taker_buy_quote),
    ]:
        if not np.all(np.isfinite(arr)):
            raise CorruptedCandleValueError(f"Non-finite float value produced in numerical array {name}")

    return NumericalCandleArrays(
        symbol=symbol,
        interval=interval,
        length=count,
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


def partition_contiguous_segments(
    candles: Sequence[Union[FeatureCandle, Dict[str, Any], Any]],
    interval: Union[str, KlineInterval],
    symbol: str = "UNKNOWN",
    server_time_ms: Optional[int] = None,
) -> List[CandleSegment]:
    """Partition a sequence of candles into contiguous segments separated by interior gaps.

    Continuous candles (delta == expected_interval_ms) belong to the same segment.
    An interior timestamp gap (delta > expected_interval_ms) terminates the current
    segment and begins a new segment.

    Zero synthetic candles are created. Original candle data is preserved.
    """
    validated = validate_candle_sequence(candles, interval=interval, server_time_ms=server_time_ms)
    iv_str = KlineInterval.from_value(interval).value
    expected_ms = interval_to_milliseconds(iv_str)

    segments: List[CandleSegment] = []
    current_segment_candles: List[FeatureCandle] = []

    for c in validated:
        if not current_segment_candles:
            current_segment_candles.append(c)
            continue

        prev_ot = current_segment_candles[-1].open_time
        delta = c.open_time - prev_ot

        if delta == expected_ms:
            current_segment_candles.append(c)
        elif delta > expected_ms:
            # Interior gap detected: close existing segment and start new segment
            seg_candles = tuple(current_segment_candles)
            segments.append(
                CandleSegment(
                    symbol=symbol,
                    interval=iv_str,
                    candles=seg_candles,
                    start_open_time=seg_candles[0].open_time,
                    end_open_time=seg_candles[-1].open_time,
                    length=len(seg_candles),
                )
            )
            current_segment_candles = [c]
        else:
            # Delta < expected_ms is caught by validate_candle_sequence, but assert here for safety
            raise MalformedIntervalSpacingError(
                f"delta {delta} ms < expected {expected_ms} ms for {iv_str}"
            )

    if current_segment_candles:
        seg_candles = tuple(current_segment_candles)
        segments.append(
            CandleSegment(
                symbol=symbol,
                interval=iv_str,
                candles=seg_candles,
                start_open_time=seg_candles[0].open_time,
                end_open_time=seg_candles[-1].open_time,
                length=len(seg_candles),
            )
        )

    return segments
