"""Chronological gap detection and continuity analysis for historical klines."""

from typing import Any, Dict, List, Optional, Sequence, Union

from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.binance.models import Kline
from trade_intelligence.klines.types import GapReport, KlineGap

INTERVAL_MS: Dict[str, int] = {
    KlineInterval.INTERVAL_5M.value: 5 * 60 * 1000,
    KlineInterval.INTERVAL_15M.value: 15 * 60 * 1000,
    KlineInterval.INTERVAL_1H.value: 60 * 60 * 1000,
    KlineInterval.INTERVAL_4H.value: 4 * 60 * 60 * 1000,
    KlineInterval.INTERVAL_1D.value: 24 * 60 * 60 * 1000,
}


def interval_to_milliseconds(interval: Union[str, KlineInterval]) -> int:
    """Return interval duration in milliseconds.

    Args:
        interval: Kline interval string or enum.

    Returns:
        Integer duration in milliseconds.

    Raises:
        ValueError: If interval is unsupported.
    """
    iv = KlineInterval.from_value(interval).value
    if iv not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: '{interval}'")
    return INTERVAL_MS[iv]


def detect_kline_gaps(
    symbol: str,
    interval: Union[str, KlineInterval],
    klines: Sequence[Union[Kline, Dict[str, Any]]],
    expected_start_time: Optional[int] = None,
    expected_end_time: Optional[int] = None,
) -> GapReport:
    """Analyze continuity of an examined sequence of klines and report gaps.

    Note:
        This analysis strictly reports continuity across the provided sequence
        of klines (i.e. the sequence examined during a download operation). It
        does not perform a full historical scan of prior uninspected database records.

    Args:
        symbol: Trading pair symbol.
        interval: Kline interval (e.g. '1h').
        klines: Chronological sequence of Kline instances or dicts containing 'open_time'.
        expected_start_time: Optional inclusive start timestamp bound.
        expected_end_time: Optional inclusive end timestamp bound for open_time.

    Returns:
        GapReport with detected gaps, expected/actual counts, and coverage ratio.
    """
    if not isinstance(symbol, str):
        raise TypeError(f"symbol must be a string, got {type(symbol).__name__}")
    if not symbol.strip():
        raise ValueError("symbol cannot be empty or whitespace only")

    if expected_start_time is not None:
        if isinstance(expected_start_time, bool) or not isinstance(expected_start_time, int):
            raise TypeError(f"expected_start_time must be an integer, got {type(expected_start_time).__name__}")
        if expected_start_time < 0:
            raise ValueError(f"expected_start_time must be non-negative, got {expected_start_time}")

    if expected_end_time is not None:
        if isinstance(expected_end_time, bool) or not isinstance(expected_end_time, int):
            raise TypeError(f"expected_end_time must be an integer, got {type(expected_end_time).__name__}")
        if expected_end_time < 0:
            raise ValueError(f"expected_end_time must be non-negative, got {expected_end_time}")

    if expected_start_time is not None and expected_end_time is not None and expected_start_time > expected_end_time:
        raise ValueError(
            f"expected_start_time ({expected_start_time}) cannot be greater than expected_end_time ({expected_end_time})"
        )

    iv_str = KlineInterval.from_value(interval).value
    interval_ms = interval_to_milliseconds(iv_str)

    # Extract open_times
    open_times: List[int] = []
    for k in klines:
        if isinstance(k, dict):
            val = k.get("open_time") if "open_time" in k else k.get("open_time_ms")
            if val is None:
                raise ValueError("Dict kline missing open_time or open_time_ms")
            ts = int(val)
            if ts < 0:
                raise ValueError(f"kline open_time must be non-negative, got: {ts}")
            open_times.append(ts)
        elif hasattr(k, "open_time_ms"):
            ts = int(k.open_time_ms)
            if ts < 0:
                raise ValueError(f"kline open_time_ms must be non-negative, got: {ts}")
            open_times.append(ts)
        elif hasattr(k, "open_time"):
            ts = int(k.open_time)
            if ts < 0:
                raise ValueError(f"kline open_time must be non-negative, got: {ts}")
            open_times.append(ts)
        else:
            raise TypeError(f"Unsupported kline item type: {type(k).__name__}")

    open_times = sorted(set(open_times))

    gaps: List[KlineGap] = []

    if not open_times:
        if expected_start_time is not None and expected_end_time is not None and expected_end_time >= expected_start_time:
            # Entire expected range is missing
            expected_count = ((expected_end_time - expected_start_time) // interval_ms) + 1
            if expected_count > 0:
                gaps.append(
                    KlineGap(
                        symbol=symbol,
                        interval=iv_str,
                        gap_start_ms=expected_start_time,
                        gap_end_ms=expected_start_time + (expected_count * interval_ms) - 1,
                        missing_candles=expected_count,
                    )
                )
                return GapReport(
                    symbol=symbol,
                    interval=iv_str,
                    total_expected_candles=expected_count,
                    total_actual_candles=0,
                    total_missing_candles=expected_count,
                    coverage_ratio=0.0,
                    gaps=gaps,
                )
        return GapReport(
            symbol=symbol,
            interval=iv_str,
            total_expected_candles=0,
            total_actual_candles=0,
            total_missing_candles=0,
            coverage_ratio=1.0,
            gaps=[],
        )

    # Check leading gap if expected_start_time was provided
    if expected_start_time is not None and open_times[0] > expected_start_time:
        delta_lead = open_times[0] - expected_start_time
        missing_lead = delta_lead // interval_ms
        if missing_lead > 0:
            gaps.append(
                KlineGap(
                    symbol=symbol,
                    interval=iv_str,
                    gap_start_ms=expected_start_time,
                    gap_end_ms=expected_start_time + (missing_lead * interval_ms) - 1,
                    missing_candles=missing_lead,
                )
            )

    # Check interior gaps
    for i in range(len(open_times) - 1):
        delta = open_times[i + 1] - open_times[i]
        if delta > interval_ms:
            missing_count = (delta // interval_ms) - 1
            if missing_count > 0:
                gap_start = open_times[i] + interval_ms
                gap_end = open_times[i + 1] - 1
                gaps.append(
                    KlineGap(
                        symbol=symbol,
                        interval=iv_str,
                        gap_start_ms=gap_start,
                        gap_end_ms=gap_end,
                        missing_candles=missing_count,
                    )
                )

    # Check trailing gap if expected_end_time was provided
    if expected_end_time is not None:
        last_open = open_times[-1]
        next_expected_open = last_open + interval_ms
        if next_expected_open <= expected_end_time:
            missing_trail = ((expected_end_time - next_expected_open) // interval_ms) + 1
            if missing_trail > 0:
                gaps.append(
                    KlineGap(
                        symbol=symbol,
                        interval=iv_str,
                        gap_start_ms=next_expected_open,
                        gap_end_ms=next_expected_open + (missing_trail * interval_ms) - 1,
                        missing_candles=missing_trail,
                    )
                )

    total_actual = len(open_times)
    total_missing = sum(g.missing_candles for g in gaps)
    total_expected = total_actual + total_missing
    coverage_ratio = (total_actual / total_expected) if total_expected > 0 else 1.0

    return GapReport(
        symbol=symbol,
        interval=iv_str,
        total_expected_candles=total_expected,
        total_actual_candles=total_actual,
        total_missing_candles=total_missing,
        coverage_ratio=round(coverage_ratio, 6),
        gaps=gaps,
    )
