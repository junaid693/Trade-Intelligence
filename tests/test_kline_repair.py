"""Tests for GapRepairer: segment planning, execution, and verification."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.binance.exceptions import BinanceConnectionError
from trade_intelligence.db.database import Database
from trade_intelligence.klines.downloader import HistoricalKlineDownloader
from trade_intelligence.klines.gap_detector import INTERVAL_MS
from trade_intelligence.klines.pipeline.coverage import CoverageScanner
from trade_intelligence.klines.pipeline.repair import GapRepairer
from trade_intelligence.klines.pipeline.types import (
    CoverageGap,
    CoverageReport,
    GapType,
    RepairStatus,
)
from trade_intelligence.klines.types import DownloadResult, GapReport


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIVE_MIN = INTERVAL_MS[KlineInterval.INTERVAL_5M.value]
HOUR = INTERVAL_MS[KlineInterval.INTERVAL_1H.value]
DAY = INTERVAL_MS[KlineInterval.INTERVAL_1D.value]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def db(db_path):
    database = Database(db_path)
    database.connect()
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def scanner(db):
    return CoverageScanner(db)


def _make_download_result(symbol="BTCUSDT", interval="1h", stored=0):
    """Create a minimal DownloadResult for mocking."""
    return DownloadResult(
        symbol=symbol,
        interval=interval,
        run_id=1,
        requested_start_time=0,
        requested_end_time=HOUR,
        effective_start_time=0,
        effective_end_time=HOUR,
        pages_fetched=1,
        records_fetched=stored,
        records_stored=stored,
        gap_report=GapReport(
            symbol=symbol,
            interval=interval,
            total_expected_candles=stored,
            total_actual_candles=stored,
            total_missing_candles=0,
            coverage_ratio=1.0,
        ),
        execution_time_seconds=0.1,
        status="completed",
    )


def _make_coverage_report_with_gaps():
    """CoverageReport with a leading gap, interior gap, and trailing gap."""
    leading = CoverageGap(
        symbol="BTCUSDT", interval="1h", gap_type=GapType.LEADING,
        gap_start_open_time_ms=0, gap_end_open_time_ms=2 * HOUR,
        missing_candles=3,
    )
    interior = CoverageGap(
        symbol="BTCUSDT", interval="1h", gap_type=GapType.INTERIOR,
        gap_start_open_time_ms=5 * HOUR, gap_end_open_time_ms=6 * HOUR,
        missing_candles=2,
    )
    trailing = CoverageGap(
        symbol="BTCUSDT", interval="1h", gap_type=GapType.TRAILING,
        gap_start_open_time_ms=9 * HOUR, gap_end_open_time_ms=10 * HOUR,
        missing_candles=2,
    )
    return CoverageReport(
        symbol="BTCUSDT", interval="1h",
        requested_start_ms=0, requested_end_ms=10 * HOUR,
        aligned_start_ms=0, aligned_end_ms=10 * HOUR,
        expected_candles=11, actual_candles=4, missing_candles=7,
        coverage_ratio=0.363636, is_complete=False,
        leading_gaps=[leading], interior_gaps=[interior], trailing_gaps=[trailing],
        all_gaps=[leading, interior, trailing],
    )


# ---------------------------------------------------------------------------
# Segment planning tests
# ---------------------------------------------------------------------------

class TestPlanRepairSegments:
    def test_all_gaps_produce_segments(self):
        report = _make_coverage_report_with_gaps()
        segments = GapRepairer.plan_repair_segments(report)

        assert len(segments) == 3
        assert segments[0].segment_start_open_time_ms == 0
        assert segments[0].segment_end_open_time_ms == 2 * HOUR
        assert segments[0].expected_missing_candles == 3

        assert segments[1].segment_start_open_time_ms == 5 * HOUR
        assert segments[1].segment_end_open_time_ms == 6 * HOUR

        assert segments[2].segment_start_open_time_ms == 9 * HOUR
        assert segments[2].segment_end_open_time_ms == 10 * HOUR

    def test_interior_only_filters(self):
        report = _make_coverage_report_with_gaps()
        segments = GapRepairer.plan_repair_segments(report, interior_only=True)

        assert len(segments) == 1
        assert segments[0].source_gap.gap_type == GapType.INTERIOR

    def test_complete_report_produces_no_segments(self):
        report = CoverageReport(
            symbol="BTCUSDT", interval="1h",
            requested_start_ms=0, requested_end_ms=5 * HOUR,
            aligned_start_ms=0, aligned_end_ms=5 * HOUR,
            expected_candles=6, actual_candles=6, missing_candles=0,
            coverage_ratio=1.0, is_complete=True,
        )
        segments = GapRepairer.plan_repair_segments(report)
        assert segments == []

    def test_disjoint_gaps_not_merged(self):
        """Each gap produces an independent segment — no merging."""
        report = _make_coverage_report_with_gaps()
        segments = GapRepairer.plan_repair_segments(report)

        # Verify each segment maps 1:1 to its source gap
        for seg in segments:
            assert seg.segment_start_open_time_ms == seg.source_gap.gap_start_open_time_ms
            assert seg.segment_end_open_time_ms == seg.source_gap.gap_end_open_time_ms

    def test_invalid_report_type_raises(self):
        with pytest.raises(TypeError, match="CoverageReport"):
            GapRepairer.plan_repair_segments("not_a_report")

    def test_full_range_gap_produces_segment(self):
        gap = CoverageGap(
            symbol="BTCUSDT", interval="1h", gap_type=GapType.FULL_RANGE,
            gap_start_open_time_ms=0, gap_end_open_time_ms=5 * HOUR,
            missing_candles=6,
        )
        report = CoverageReport(
            symbol="BTCUSDT", interval="1h",
            requested_start_ms=0, requested_end_ms=5 * HOUR,
            aligned_start_ms=0, aligned_end_ms=5 * HOUR,
            expected_candles=6, actual_candles=0, missing_candles=6,
            coverage_ratio=0.0, is_complete=False,
            leading_gaps=[gap], all_gaps=[gap],
        )
        segments = GapRepairer.plan_repair_segments(report)
        assert len(segments) == 1
        assert segments[0].expected_missing_candles == 6


# ---------------------------------------------------------------------------
# Segment execution tests
# ---------------------------------------------------------------------------

class TestExecuteRepairs:
    def test_successful_repair(self):
        report = _make_coverage_report_with_gaps()
        segments = GapRepairer.plan_repair_segments(report, interior_only=True)
        assert len(segments) == 1

        mock_dl = MagicMock(spec=HistoricalKlineDownloader)
        mock_dl.download_historical_klines.return_value = _make_download_result(stored=2)

        results = GapRepairer.execute_repairs(segments, mock_dl)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].candles_recovered == 2
        assert results[0].error is None

        # Verify boundary translation
        call_args = mock_dl.download_historical_klines.call_args
        assert call_args.kwargs["start_time"] == 5 * HOUR
        assert call_args.kwargs["end_time"] == 6 * HOUR + HOUR - 1
        assert call_args.kwargs["resume"] is False

    def test_failed_repair_continues(self):
        """Downloader exception → recorded as failure; surrounding data preserved."""
        report = _make_coverage_report_with_gaps()
        segments = GapRepairer.plan_repair_segments(report)
        assert len(segments) == 3

        mock_dl = MagicMock(spec=HistoricalKlineDownloader)
        mock_dl.download_historical_klines.side_effect = [
            _make_download_result(stored=3),      # 1st segment succeeds
            BinanceConnectionError("timeout"),     # 2nd segment fails
            _make_download_result(stored=2),       # 3rd segment succeeds
        ]

        results = GapRepairer.execute_repairs(segments, mock_dl)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error == "timeout"
        assert results[1].download_result is None
        assert results[2].success is True

    def test_zero_retry_logic(self):
        """GapRepairer does NOT retry — a single exception is immediately recorded."""
        report = _make_coverage_report_with_gaps()
        segments = GapRepairer.plan_repair_segments(report, interior_only=True)

        mock_dl = MagicMock(spec=HistoricalKlineDownloader)
        mock_dl.download_historical_klines.side_effect = RuntimeError("fatal")

        results = GapRepairer.execute_repairs(segments, mock_dl)

        assert len(results) == 1
        assert results[0].success is False
        # Only one call to the downloader — no retry
        assert mock_dl.download_historical_klines.call_count == 1

    def test_invalid_downloader_type_raises(self):
        with pytest.raises(TypeError, match="HistoricalKlineDownloader"):
            GapRepairer.execute_repairs([], "not_a_downloader")

    def test_empty_segments_returns_empty(self):
        mock_dl = MagicMock(spec=HistoricalKlineDownloader)
        results = GapRepairer.execute_repairs([], mock_dl)
        assert results == []
        mock_dl.download_historical_klines.assert_not_called()

    def test_idempotent_repair_no_new_downloads(self):
        """Re-running repair on a complete range results in no segments."""
        report = CoverageReport(
            symbol="BTCUSDT", interval="1h",
            requested_start_ms=0, requested_end_ms=5 * HOUR,
            aligned_start_ms=0, aligned_end_ms=5 * HOUR,
            expected_candles=6, actual_candles=6, missing_candles=0,
            coverage_ratio=1.0, is_complete=True,
        )
        segments = GapRepairer.plan_repair_segments(report)
        assert segments == []

        mock_dl = MagicMock(spec=HistoricalKlineDownloader)
        results = GapRepairer.execute_repairs(segments, mock_dl)
        assert results == []
        mock_dl.download_historical_klines.assert_not_called()


# ---------------------------------------------------------------------------
# Boundary translation audit & regression tests
# ---------------------------------------------------------------------------

class TestRepairBoundaryTranslation:
    """Audit and prove exact boundary translation for repair requests across timeframes."""

    @pytest.mark.parametrize(
        "interval_str,interval_ms",
        [
            ("5m", FIVE_MIN),
            ("1h", HOUR),
            ("1d", DAY),
        ],
    )
    def test_boundary_translation_exactness(self, interval_str, interval_ms):
        """Prove that missing open-time candles [t0, t1, t2] result in a repair request
        that includes exactly those candles, includes t2, excludes t3, and excludes preceding candle.
        Specifically verifies:
          Missing: 12:00, 13:00, 14:00 (for 1h, and corresponding intervals for 5m, 1d)
          Repair request includes: 12:00, 13:00, 14:00
          Repair request excludes: 11:00 (preceding) and 15:00 (next adjacent).
        """
        # Anchor timestamps
        t_prev = 11 * interval_ms  # preceding candle (e.g. 11:00)
        t0 = 12 * interval_ms      # 1st missing (e.g. 12:00)
        t1 = 13 * interval_ms      # 2nd missing (e.g. 13:00)
        t2 = 14 * interval_ms      # 3rd missing (e.g. 14:00)
        t3 = 15 * interval_ms      # next adjacent candle (e.g. 15:00)

        gap = CoverageGap(
            symbol="BTCUSDT",
            interval=interval_str,
            gap_type=GapType.INTERIOR,
            gap_start_open_time_ms=t0,
            gap_end_open_time_ms=t2,
            missing_candles=3,
        )
        report = CoverageReport(
            symbol="BTCUSDT",
            interval=interval_str,
            requested_start_ms=10 * interval_ms,
            requested_end_ms=16 * interval_ms,
            aligned_start_ms=10 * interval_ms,
            aligned_end_ms=16 * interval_ms,
            expected_candles=7,
            actual_candles=4,
            missing_candles=3,
            coverage_ratio=4 / 7,
            is_complete=False,
            interior_gaps=[gap],
            all_gaps=[gap],
        )

        segments = GapRepairer.plan_repair_segments(report)
        assert len(segments) == 1
        seg = segments[0]

        # Verify RepairSegment holds exact inclusive open_times
        assert seg.segment_start_open_time_ms == t0
        assert seg.segment_end_open_time_ms == t2
        assert seg.expected_missing_candles == 3

        mock_dl = MagicMock(spec=HistoricalKlineDownloader)
        mock_dl.download_historical_klines.return_value = _make_download_result(
            symbol="BTCUSDT", interval=interval_str, stored=3
        )

        results = GapRepairer.execute_repairs(segments, mock_dl)
        assert len(results) == 1
        assert results[0].success is True

        mock_dl.download_historical_klines.assert_called_once()
        call_kwargs = mock_dl.download_historical_klines.call_args.kwargs
        req_start = call_kwargs["start_time"]
        req_end = call_kwargs["end_time"]

        # Exact boundary assertions
        assert req_start == t0
        assert req_end == t2 + interval_ms - 1
        assert call_kwargs["resume"] is False

        # Invariant 1: Preceding candle (11:00 / t_prev) is strictly BEFORE request start
        assert t_prev < req_start, f"{t_prev} must be < {req_start}"

        # Invariant 2: First missing candle (12:00 / t0) is covered
        assert req_start <= t0 <= req_end

        # Invariant 3: Intermediate missing candle (13:00 / t1) is covered
        assert req_start <= t1 <= req_end

        # Invariant 4: Last missing candle (14:00 / t2) is strictly covered (NOT omitted)
        assert req_start <= t2 <= req_end

        # Invariant 5: Next candle (15:00 / t3) is strictly AFTER request end (NOT included)
        assert t3 > req_end, f"{t3} must be strictly > {req_end} (next candle must NOT be requested)"

        # Invariant 6: The request end window does not touch or exceed the next candle's open time
        assert req_end < t3


# ---------------------------------------------------------------------------
# Repair verification tests
# ---------------------------------------------------------------------------

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
            (symbol, interval, ot, ot + HOUR - 1),
        )
    conn.commit()


class TestVerifyRepairs:
    def test_verified_complete(self, db, scanner):
        _seed_symbol(db)
        open_times = [i * HOUR for i in range(6)]
        _seed_klines(db, "BTCUSDT", "1h", open_times)

        report, status = GapRepairer.verify_repairs(
            "BTCUSDT", "1h", 0, 5 * HOUR, scanner
        )

        assert report.is_complete is True
        assert status == RepairStatus.COMPLETED

    def test_verified_incomplete(self, db, scanner):
        _seed_symbol(db)
        # Missing hour 3
        open_times = [0, HOUR, 2 * HOUR, 4 * HOUR, 5 * HOUR]
        _seed_klines(db, "BTCUSDT", "1h", open_times)

        report, status = GapRepairer.verify_repairs(
            "BTCUSDT", "1h", 0, 5 * HOUR, scanner
        )

        assert report.is_complete is False
        assert status == RepairStatus.INCOMPLETE
        assert len(report.interior_gaps) == 1
