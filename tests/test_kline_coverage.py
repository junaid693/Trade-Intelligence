"""Tests for CoverageScanner and coverage utility functions."""

import os
import tempfile

import pytest

from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.db.database import Database
from trade_intelligence.klines.gap_detector import INTERVAL_MS
from trade_intelligence.klines.pipeline.coverage import (
    CoverageScanner,
    align_end_time,
    align_start_time,
    compute_expected_count,
)
from trade_intelligence.klines.pipeline.types import GapType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path():
    """Create a temporary database path that is cleaned up after the test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def db(db_path):
    """Initialised Database instance with schema."""
    database = Database(db_path)
    database.connect()
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def scanner(db):
    return CoverageScanner(db)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HOUR = INTERVAL_MS[KlineInterval.INTERVAL_1H.value]
FOUR_HOUR = INTERVAL_MS[KlineInterval.INTERVAL_4H.value]
DAY = INTERVAL_MS[KlineInterval.INTERVAL_1D.value]
FIVE_MIN = INTERVAL_MS[KlineInterval.INTERVAL_5M.value]
FIFTEEN_MIN = INTERVAL_MS[KlineInterval.INTERVAL_15M.value]


def _seed_symbol(db, symbol="BTCUSDT"):
    """Insert a symbol into the symbols table for FK protection."""
    db.insert_symbol(
        symbol=symbol,
        base_asset=symbol[:-4] if symbol.endswith("USDT") else "BTC",
        quote_asset="USDT",
        status="TRADING",
        is_spot_trading_allowed=True,
        is_margin_trading_allowed=True,
        base_asset_precision=8,
        quote_asset_precision=8,
        updated_at=1700000000000,
    )


def _seed_klines(db, symbol, interval, open_times):
    """Seed klines with minimal data for coverage scanning."""
    conn = db.connection
    for ot in open_times:
        conn.execute(
            """
            INSERT OR IGNORE INTO klines
                (symbol, interval, open_time, open_price, high_price, low_price, close_price, volume,
                 close_time, quote_asset_volume, number_of_trades, taker_buy_base_volume,
                 taker_buy_quote_volume, raw_response_id)
            VALUES (?, ?, ?, '100.0', '101.0', '99.0', '100.5', '10.0',
                    ?, '1000.0', 100, '5.0', '500.0', NULL)
            """,
            (symbol, interval, ot, ot + INTERVAL_MS.get(interval, HOUR) - 1),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Alignment tests
# ---------------------------------------------------------------------------

class TestAlignStartTime:
    def test_already_aligned(self):
        assert align_start_time(0, HOUR) == 0
        assert align_start_time(HOUR, HOUR) == HOUR

    def test_unaligned(self):
        assert align_start_time(HOUR + 500, HOUR) == HOUR

    def test_daily(self):
        assert align_start_time(DAY + 1000, DAY) == DAY


class TestAlignEndTime:
    def test_already_aligned(self):
        assert align_end_time(5 * HOUR, HOUR) == 5 * HOUR

    def test_unaligned(self):
        assert align_end_time(5 * HOUR + 999, HOUR) == 5 * HOUR


class TestComputeExpectedCount:
    def test_single_candle(self):
        assert compute_expected_count(0, 0, HOUR) == 1

    def test_two_candles(self):
        assert compute_expected_count(0, HOUR, HOUR) == 2

    def test_five_hourly(self):
        assert compute_expected_count(0, 4 * HOUR, HOUR) == 5

    def test_daily(self):
        assert compute_expected_count(0, 6 * DAY, DAY) == 7

    def test_inverted_returns_zero(self):
        assert compute_expected_count(HOUR, 0, HOUR) == 0

    def test_zero_interval_returns_zero(self):
        assert compute_expected_count(0, HOUR, 0) == 0

    def test_all_intervals(self):
        """Verify expected counts for all 5 supported intervals."""
        intervals = [
            (FIVE_MIN, 12 + 1),    # 12 five-minute intervals = 13 candles
            (FIFTEEN_MIN, 4 + 1),  # 4 fifteen-minute intervals = 5 candles
            (HOUR, 1 + 1),         # 1 hour interval = 2 candles
            (FOUR_HOUR, 0 + 1),    # less than 4h = 1 candle (start == end after align)
            (DAY, 0 + 1),          # less than 1d = 1 candle
        ]
        for ms, expected in intervals:
            # 1 hour range
            result = compute_expected_count(0, HOUR, ms)
            assert result == expected, f"interval_ms={ms}: expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# Coverage scan tests
# ---------------------------------------------------------------------------

class TestCoverageScanComplete:
    """Tests where all expected candles are present."""

    def test_complete_hourly_sequence(self, db, scanner):
        _seed_symbol(db)
        open_times = [i * HOUR for i in range(10)]
        _seed_klines(db, "BTCUSDT", "1h", open_times)

        report = scanner.scan_coverage("BTCUSDT", "1h", 0, 9 * HOUR)

        assert report.is_complete is True
        assert report.expected_candles == 10
        assert report.actual_candles == 10
        assert report.missing_candles == 0
        assert report.coverage_ratio == 1.0
        assert len(report.all_gaps) == 0

    def test_complete_daily_sequence(self, db, scanner):
        _seed_symbol(db)
        open_times = [i * DAY for i in range(5)]
        _seed_klines(db, "BTCUSDT", "1d", open_times)

        report = scanner.scan_coverage("BTCUSDT", "1d", 0, 4 * DAY)

        assert report.is_complete is True
        assert report.expected_candles == 5
        assert report.actual_candles == 5


class TestCoverageScanLeadingGap:
    def test_leading_gap(self, db, scanner):
        _seed_symbol(db)
        # Candles start at hour 3, but range starts at hour 0
        open_times = [3 * HOUR, 4 * HOUR, 5 * HOUR]
        _seed_klines(db, "BTCUSDT", "1h", open_times)

        report = scanner.scan_coverage("BTCUSDT", "1h", 0, 5 * HOUR)

        assert report.is_complete is False
        assert report.expected_candles == 6
        assert report.actual_candles == 3
        assert report.missing_candles == 3
        assert len(report.leading_gaps) == 1
        gap = report.leading_gaps[0]
        assert gap.gap_type == GapType.LEADING
        assert gap.gap_start_open_time_ms == 0
        assert gap.gap_end_open_time_ms == 2 * HOUR
        assert gap.missing_candles == 3


class TestCoverageScanInteriorGap:
    def test_single_interior_gap(self, db, scanner):
        _seed_symbol(db)
        # Hours: 0, 1, 2, [gap: 3, 4], 5, 6
        open_times = [0, HOUR, 2 * HOUR, 5 * HOUR, 6 * HOUR]
        _seed_klines(db, "BTCUSDT", "1h", open_times)

        report = scanner.scan_coverage("BTCUSDT", "1h", 0, 6 * HOUR)

        assert report.is_complete is False
        assert len(report.interior_gaps) == 1
        gap = report.interior_gaps[0]
        assert gap.gap_type == GapType.INTERIOR
        assert gap.gap_start_open_time_ms == 3 * HOUR
        assert gap.gap_end_open_time_ms == 4 * HOUR
        assert gap.missing_candles == 2

    def test_multiple_interior_gaps(self, db, scanner):
        _seed_symbol(db)
        # Hours: 0, 1, [gap: 2, 3], 4, [gap: 5, 6, 7], 8
        open_times = [0, HOUR, 4 * HOUR, 8 * HOUR]
        _seed_klines(db, "BTCUSDT", "1h", open_times)

        report = scanner.scan_coverage("BTCUSDT", "1h", 0, 8 * HOUR)

        assert len(report.interior_gaps) == 2
        assert report.interior_gaps[0].missing_candles == 2
        assert report.interior_gaps[1].missing_candles == 3


class TestCoverageScanTrailingGap:
    def test_trailing_gap(self, db, scanner):
        _seed_symbol(db)
        # Candles exist for hours 0-4, but range extends to hour 7
        open_times = [i * HOUR for i in range(5)]
        _seed_klines(db, "BTCUSDT", "1h", open_times)

        report = scanner.scan_coverage("BTCUSDT", "1h", 0, 7 * HOUR)

        assert report.is_complete is False
        assert len(report.trailing_gaps) == 1
        gap = report.trailing_gaps[0]
        assert gap.gap_type == GapType.TRAILING
        assert gap.gap_start_open_time_ms == 5 * HOUR
        assert gap.gap_end_open_time_ms == 7 * HOUR
        assert gap.missing_candles == 3


class TestCoverageScanFullRange:
    def test_no_candles_at_all(self, db, scanner):
        _seed_symbol(db)
        report = scanner.scan_coverage("BTCUSDT", "1h", 0, 5 * HOUR)

        assert report.is_complete is False
        assert report.actual_candles == 0
        assert report.expected_candles == 6
        assert report.missing_candles == 6
        assert len(report.all_gaps) == 1
        assert report.all_gaps[0].gap_type == GapType.FULL_RANGE


class TestCoverageScanFormingCandle:
    def test_server_time_clamping(self, db, scanner):
        _seed_symbol(db)
        # 10 hourly candles 0..9H
        open_times = [i * HOUR for i in range(10)]
        _seed_klines(db, "BTCUSDT", "1h", open_times)

        # Server time is at hour 8 + 30 min → last closed open_time = 7H
        server_time_ms = 8 * HOUR + (30 * 60 * 1000)

        report = scanner.scan_coverage(
            "BTCUSDT", "1h", 0, 9 * HOUR, server_time_ms=server_time_ms
        )

        # Aligned end should be 7H (8H - 1H = 7H is the last closed candle open_time)
        assert report.aligned_end_ms == 7 * HOUR
        assert report.expected_candles == 8
        assert report.is_complete is True


class TestCoverageScanDuplicates:
    def test_duplicate_open_times_deduplicated(self, db, scanner):
        _seed_symbol(db)
        open_times = [0, HOUR, 2 * HOUR]
        _seed_klines(db, "BTCUSDT", "1h", open_times)

        report = scanner.scan_coverage("BTCUSDT", "1h", 0, 2 * HOUR)
        assert report.actual_candles == 3
        assert report.is_complete is True


class TestCoverageScanValidation:
    def test_bool_start_time_rejected(self, scanner):
        with pytest.raises(TypeError, match="start_time must be int"):
            scanner.scan_coverage("BTCUSDT", "1h", True, 1000)

    def test_bool_end_time_rejected(self, scanner):
        with pytest.raises(TypeError, match="end_time must be int"):
            scanner.scan_coverage("BTCUSDT", "1h", 0, False)

    def test_negative_start_time(self, scanner):
        with pytest.raises(ValueError, match="non-negative"):
            scanner.scan_coverage("BTCUSDT", "1h", -1, 1000)

    def test_negative_end_time(self, scanner):
        with pytest.raises(ValueError, match="non-negative"):
            scanner.scan_coverage("BTCUSDT", "1h", 0, -1)

    def test_start_greater_than_end(self, scanner):
        with pytest.raises(ValueError, match="start_time.*>.*end_time"):
            scanner.scan_coverage("BTCUSDT", "1h", 5000, 1000)

    def test_invalid_interval(self, scanner):
        with pytest.raises(ValueError, match="Unsupported"):
            scanner.scan_coverage("BTCUSDT", "2h", 0, 1000)

    def test_empty_symbol(self, scanner):
        with pytest.raises(ValueError, match="non-empty"):
            scanner.scan_coverage("", "1h", 0, 1000)

    def test_non_database_constructor(self):
        with pytest.raises(TypeError, match="Database"):
            CoverageScanner("not_a_db")


class TestCoverageScanAllIntervals:
    """Verify scanning works correctly for all 5 intervals."""

    @pytest.mark.parametrize(
        "interval_str,interval_ms",
        [
            ("5m", FIVE_MIN),
            ("15m", FIFTEEN_MIN),
            ("1h", HOUR),
            ("4h", FOUR_HOUR),
            ("1d", DAY),
        ],
    )
    def test_complete_sequence_each_interval(self, db, scanner, interval_str, interval_ms):
        _seed_symbol(db)
        open_times = [i * interval_ms for i in range(5)]
        _seed_klines(db, "BTCUSDT", interval_str, open_times)

        report = scanner.scan_coverage("BTCUSDT", interval_str, 0, 4 * interval_ms)

        assert report.is_complete is True
        assert report.expected_candles == 5
        assert report.actual_candles == 5
        assert report.missing_candles == 0
