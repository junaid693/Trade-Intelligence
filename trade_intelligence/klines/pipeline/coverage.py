"""Full-range coverage scanning and continuity analysis for historical klines."""

import logging
from typing import List, Optional, Union

from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.db.database import Database
from trade_intelligence.klines.gap_detector import interval_to_milliseconds
from trade_intelligence.klines.pipeline.types import (
    CoverageGap,
    CoverageReport,
    GapType,
)

logger = logging.getLogger(__name__)


def align_start_time(start_time: int, interval_ms: int) -> int:
    """Align a timestamp down to the nearest interval boundary.

    Args:
        start_time: UTC epoch milliseconds.
        interval_ms: Interval duration in milliseconds.

    Returns:
        Aligned start time (floor).
    """
    return start_time - (start_time % interval_ms)


def align_end_time(end_time: int, interval_ms: int) -> int:
    """Align a timestamp down to the nearest interval boundary.

    For coverage purposes the *end* represents the open_time of the last
    expected candle, so we align downward just like the start.

    Args:
        end_time: UTC epoch milliseconds.
        interval_ms: Interval duration in milliseconds.

    Returns:
        Aligned end time (floor).
    """
    return end_time - (end_time % interval_ms)


def compute_expected_count(aligned_start: int, aligned_end: int, interval_ms: int) -> int:
    """Compute the number of expected candles between two aligned boundaries (inclusive).

    Args:
        aligned_start: Aligned start open_time (inclusive).
        aligned_end: Aligned end open_time (inclusive).
        interval_ms: Interval duration in milliseconds.

    Returns:
        Non-negative integer count.
    """
    if aligned_end < aligned_start or interval_ms <= 0:
        return 0
    return ((aligned_end - aligned_start) // interval_ms) + 1


class CoverageScanner:
    """Scans the database for candle coverage over a requested range."""

    def __init__(self, db: Database) -> None:
        if not isinstance(db, Database):
            raise TypeError(f"db must be a Database instance, got {type(db).__name__}")
        self._db = db

    def scan_coverage(
        self,
        symbol: str,
        interval: Union[str, KlineInterval],
        start_time: int,
        end_time: int,
        server_time_ms: Optional[int] = None,
    ) -> CoverageReport:
        """Analyse candle coverage for a specific (symbol, interval, range).

        Args:
            symbol: Trading pair symbol.
            interval: Kline interval.
            start_time: Requested inclusive start UTC epoch ms.
            end_time: Requested inclusive end UTC epoch ms.
            server_time_ms: Optional current Binance server time.  When
                provided the effective end is clamped so that forming
                candles cannot make a range appear complete.

        Returns:
            CoverageReport with categorised gaps and coverage statistics.

        Raises:
            TypeError / ValueError: On invalid inputs.
        """
        # --- validation ---
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        sym = symbol.strip().upper()

        validated_interval = KlineInterval.from_value(interval)
        iv_str = validated_interval.value
        interval_ms = interval_to_milliseconds(validated_interval)

        if isinstance(start_time, bool) or not isinstance(start_time, int):
            raise TypeError(f"start_time must be int, got {type(start_time).__name__}")
        if start_time < 0:
            raise ValueError(f"start_time must be non-negative, got {start_time}")
        if isinstance(end_time, bool) or not isinstance(end_time, int):
            raise TypeError(f"end_time must be int, got {type(end_time).__name__}")
        if end_time < 0:
            raise ValueError(f"end_time must be non-negative, got {end_time}")
        if start_time > end_time:
            raise ValueError(f"start_time ({start_time}) > end_time ({end_time})")

        # --- alignment ---
        aligned_start = align_start_time(start_time, interval_ms)
        aligned_end = align_end_time(end_time, interval_ms)

        # Clamp for forming candles
        if server_time_ms is not None:
            max_closed_open = align_start_time(server_time_ms, interval_ms) - interval_ms
            if max_closed_open < aligned_end:
                aligned_end = max_closed_open

        if aligned_end < aligned_start:
            return CoverageReport(
                symbol=sym,
                interval=iv_str,
                requested_start_ms=start_time,
                requested_end_ms=end_time,
                aligned_start_ms=aligned_start,
                aligned_end_ms=aligned_end,
                expected_candles=0,
                actual_candles=0,
                missing_candles=0,
                coverage_ratio=1.0,
                is_complete=True,
            )

        expected = compute_expected_count(aligned_start, aligned_end, interval_ms)

        # --- database query ---
        open_times = self._db.query_kline_open_times_range(
            symbol=sym,
            interval=validated_interval,
            start_time=aligned_start,
            end_time=aligned_end,
        )

        # Defensive: sort and deduplicate
        open_times = sorted(set(open_times))
        actual = len(open_times)

        # --- gap detection ---
        leading: List[CoverageGap] = []
        interior: List[CoverageGap] = []
        trailing: List[CoverageGap] = []

        if actual == 0:
            # Entire range is missing
            if expected > 0:
                gap = CoverageGap(
                    symbol=sym,
                    interval=iv_str,
                    gap_type=GapType.FULL_RANGE,
                    gap_start_open_time_ms=aligned_start,
                    gap_end_open_time_ms=aligned_end,
                    missing_candles=expected,
                )
                leading.append(gap)
        else:
            # Leading gap
            if open_times[0] > aligned_start:
                missing_lead = (open_times[0] - aligned_start) // interval_ms
                if missing_lead > 0:
                    leading.append(CoverageGap(
                        symbol=sym,
                        interval=iv_str,
                        gap_type=GapType.LEADING,
                        gap_start_open_time_ms=aligned_start,
                        gap_end_open_time_ms=open_times[0] - interval_ms,
                        missing_candles=missing_lead,
                    ))

            # Interior gaps
            for i in range(len(open_times) - 1):
                delta = open_times[i + 1] - open_times[i]
                if delta > interval_ms:
                    missing_count = (delta // interval_ms) - 1
                    if missing_count > 0:
                        interior.append(CoverageGap(
                            symbol=sym,
                            interval=iv_str,
                            gap_type=GapType.INTERIOR,
                            gap_start_open_time_ms=open_times[i] + interval_ms,
                            gap_end_open_time_ms=open_times[i + 1] - interval_ms,
                            missing_candles=missing_count,
                        ))

            # Trailing gap
            if open_times[-1] < aligned_end:
                next_expected = open_times[-1] + interval_ms
                missing_trail = ((aligned_end - next_expected) // interval_ms) + 1
                if missing_trail > 0:
                    trailing.append(CoverageGap(
                        symbol=sym,
                        interval=iv_str,
                        gap_type=GapType.TRAILING,
                        gap_start_open_time_ms=next_expected,
                        gap_end_open_time_ms=aligned_end,
                        missing_candles=missing_trail,
                    ))

        all_gaps = leading + interior + trailing
        missing = sum(g.missing_candles for g in all_gaps)
        ratio = (actual / expected) if expected > 0 else 1.0

        return CoverageReport(
            symbol=sym,
            interval=iv_str,
            requested_start_ms=start_time,
            requested_end_ms=end_time,
            aligned_start_ms=aligned_start,
            aligned_end_ms=aligned_end,
            expected_candles=expected,
            actual_candles=actual,
            missing_candles=missing,
            coverage_ratio=round(ratio, 6),
            is_complete=(missing == 0 and actual == expected),
            leading_gaps=leading,
            interior_gaps=interior,
            trailing_gaps=trailing,
            all_gaps=all_gaps,
        )
