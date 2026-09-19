"""Integration tests for the historical data orchestration pipeline.

These tests verify the full chain:
  Database → CoverageScanner → GapRepairer → HistoricalKlineDownloader → Verify

Live tests (RUN_LIVE_TESTS=1) exercise the complete Binance API path.
"""

import os
import sqlite3
import tempfile
import time

import pytest

from trade_intelligence.binance.client import BinanceRestClient
from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.db.database import Database
from trade_intelligence.klines.downloader import HistoricalKlineDownloader
from trade_intelligence.klines.gap_detector import INTERVAL_MS
from trade_intelligence.klines.pipeline.coverage import CoverageScanner
from trade_intelligence.klines.pipeline.orchestrator import HistoricalDataOrchestrator
from trade_intelligence.klines.pipeline.repair import GapRepairer
from trade_intelligence.klines.pipeline.types import (
    CoverageStatus,
    DownloadStatus,
    GapType,
    PipelineConfig,
    PipelineRequest,
    PipelineResult,
    PipelineStatus,
    RepairStatus,
)
from trade_intelligence.universe.sync import UniverseSyncer


HOUR = INTERVAL_MS[KlineInterval.INTERVAL_1H.value]
DAY = INTERVAL_MS[KlineInterval.INTERVAL_1D.value]

SKIP_LIVE = not os.environ.get("RUN_LIVE_TESTS", "").strip()


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


def _seed_klines(db, symbol, interval, open_times, interval_ms=HOUR):
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
            (symbol, interval, ot, ot + interval_ms - 1),
        )
    conn.commit()


def _delete_klines(db, symbol, interval, open_times):
    """Remove specific klines to create artificial gaps."""
    conn = db.connection
    for ot in open_times:
        conn.execute(
            "DELETE FROM klines WHERE symbol = ? AND interval = ? AND open_time = ?",
            (symbol, interval, ot),
        )
    conn.commit()


def _integrity_check(db):
    """Run SQLite PRAGMA integrity_check and foreign_key_check."""
    conn = db.connection
    integrity = conn.execute("PRAGMA integrity_check;").fetchone()[0]
    assert integrity == "ok", f"integrity_check failed: {integrity}"

    fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
    assert len(fk_violations) == 0, f"FK violations: {fk_violations}"


# ---------------------------------------------------------------------------
# Offline integration tests
# ---------------------------------------------------------------------------

class TestOfflinePipelineIntegration:
    """Full pipeline integration tests using in-memory data (no Binance API)."""

    def test_scan_detect_repair_verify(self, db):
        """Complete offline cycle: seed → delete → scan → detect → repair → verify."""
        _seed_symbol(db)

        # Seed 24 hourly candles (0..23H)
        all_hours = [i * HOUR for i in range(24)]
        _seed_klines(db, "BTCUSDT", "1h", all_hours)

        # Create interior gap: delete hours 5, 6, 7
        _delete_klines(db, "BTCUSDT", "1h", [5 * HOUR, 6 * HOUR, 7 * HOUR])

        # Scan
        scanner = CoverageScanner(db)
        report = scanner.scan_coverage("BTCUSDT", "1h", 0, 23 * HOUR)

        assert report.is_complete is False
        assert report.missing_candles == 3
        assert len(report.interior_gaps) == 1
        assert report.interior_gaps[0].gap_start_open_time_ms == 5 * HOUR
        assert report.interior_gaps[0].gap_end_open_time_ms == 7 * HOUR

        # Plan
        segments = GapRepairer.plan_repair_segments(report)
        assert len(segments) == 1
        assert segments[0].expected_missing_candles == 3

        # Simulate repair by re-seeding deleted candles
        _seed_klines(db, "BTCUSDT", "1h", [5 * HOUR, 6 * HOUR, 7 * HOUR])

        # Verify
        final_report, status = GapRepairer.verify_repairs(
            "BTCUSDT", "1h", 0, 23 * HOUR, scanner
        )

        assert final_report.is_complete is True
        assert status == RepairStatus.COMPLETED
        _integrity_check(db)

    def test_multiple_interior_gaps(self, db):
        """Multiple disjoint gaps in a single range."""
        _seed_symbol(db)

        all_hours = [i * HOUR for i in range(24)]
        _seed_klines(db, "BTCUSDT", "1h", all_hours)

        # Gap 1: hours 3, 4
        # Gap 2: hours 10, 11, 12
        _delete_klines(db, "BTCUSDT", "1h", [3 * HOUR, 4 * HOUR])
        _delete_klines(db, "BTCUSDT", "1h", [10 * HOUR, 11 * HOUR, 12 * HOUR])

        scanner = CoverageScanner(db)
        report = scanner.scan_coverage("BTCUSDT", "1h", 0, 23 * HOUR)

        assert report.missing_candles == 5
        assert len(report.interior_gaps) == 2

        segments = GapRepairer.plan_repair_segments(report, interior_only=True)
        assert len(segments) == 2
        assert segments[0].expected_missing_candles == 2
        assert segments[1].expected_missing_candles == 3

    def test_full_range_gap(self, db):
        """No candles in range at all."""
        _seed_symbol(db)

        scanner = CoverageScanner(db)
        report = scanner.scan_coverage("BTCUSDT", "1h", 0, 5 * HOUR)

        assert report.actual_candles == 0
        assert report.expected_candles == 6
        assert len(report.all_gaps) == 1
        assert report.all_gaps[0].gap_type == GapType.FULL_RANGE
        assert len(report.full_range_gaps) == 1
        assert len(report.leading_gaps) == 0

        segments = GapRepairer.plan_repair_segments(report)
        assert len(segments) == 1
        assert segments[0].expected_missing_candles == 6

    def test_interior_only_filter(self, db):
        """repair_interior_only=True skips leading and trailing gaps."""
        _seed_symbol(db)

        # Seed hours 2..7 (leading gap 0,1 and trailing gap 8,9)
        _seed_klines(db, "BTCUSDT", "1h", [i * HOUR for i in range(2, 8)])
        # Delete hour 4 (interior gap)
        _delete_klines(db, "BTCUSDT", "1h", [4 * HOUR])

        scanner = CoverageScanner(db)
        report = scanner.scan_coverage("BTCUSDT", "1h", 0, 9 * HOUR)

        assert len(report.leading_gaps) == 1
        assert len(report.interior_gaps) == 1
        assert len(report.trailing_gaps) == 1

        all_segments = GapRepairer.plan_repair_segments(report, interior_only=False)
        interior_segments = GapRepairer.plan_repair_segments(report, interior_only=True)

        assert len(all_segments) == 3
        assert len(interior_segments) == 1
        assert interior_segments[0].source_gap.gap_type == GapType.INTERIOR

    def test_database_query_method(self, db):
        """Verify query_kline_open_times_range returns correct data."""
        _seed_symbol(db)
        expected = [i * HOUR for i in range(10)]
        _seed_klines(db, "BTCUSDT", "1h", expected)

        result = db.query_kline_open_times_range("BTCUSDT", "1h", 0, 9 * HOUR)
        assert result == expected

    def test_database_query_range_filter(self, db):
        """query_kline_open_times_range respects range boundaries."""
        _seed_symbol(db)
        _seed_klines(db, "BTCUSDT", "1h", [i * HOUR for i in range(20)])

        result = db.query_kline_open_times_range("BTCUSDT", "1h", 5 * HOUR, 10 * HOUR)
        assert result == [i * HOUR for i in range(5, 11)]


# ---------------------------------------------------------------------------
# Live integration tests (RUN_LIVE_TESTS=1)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP_LIVE, reason="Live tests disabled (set RUN_LIVE_TESTS=1)")
class TestLivePipelineIntegration:
    """Live tests that exercise the full Binance API → DB pipeline."""

    def test_live_orchestrator_btcusdt(self, db):
        """Download + scan + (optional repair) for BTCUSDT 1d over a short historical window."""
        with BinanceRestClient() as client:
            # Sync universe first to satisfy FK constraints
            syncer = UniverseSyncer(db=db, client=client)
            syncer.sync_spot_universe(quote_asset="USDT")

            downloader = HistoricalKlineDownloader(
                db=db, client=client, request_delay_ms=300.0
            )
            orch = HistoricalDataOrchestrator(
                db=db, client=client, downloader=downloader
            )

            # 2-day window, ~90 days ago
            now_ms = int(time.time() * 1000)
            start = now_ms - (90 * DAY)
            end = start + (2 * DAY)

            request = PipelineRequest(
                symbols=["BTCUSDT"],
                intervals=["1d"],
                start_time=start,
                end_time=end,
                download_before_scan=True,
                repair_gaps=True,
            )

            result = orch.run(request)

            assert result.overall_status in (
                PipelineStatus.COMPLETED,
                PipelineStatus.COMPLETED_WITH_GAPS,
            )

            tf = result.timeframe_results[("BTCUSDT", "1d")]
            assert tf.download_status == DownloadStatus.COMPLETED
            assert tf.stored_candles > 0

            _integrity_check(db)

    def test_live_orchestrator_multi_interval(self, db):
        """Download BTCUSDT across 1d and 1h for a 2-day window."""
        with BinanceRestClient() as client:
            syncer = UniverseSyncer(db=db, client=client)
            syncer.sync_spot_universe(quote_asset="USDT")

            downloader = HistoricalKlineDownloader(
                db=db, client=client, request_delay_ms=300.0
            )
            orch = HistoricalDataOrchestrator(
                db=db, client=client, downloader=downloader
            )

            now_ms = int(time.time() * 1000)
            start = now_ms - (90 * DAY)
            end = start + (2 * DAY)

            request = PipelineRequest(
                symbols=["BTCUSDT"],
                intervals=["1d", "1h"],
                start_time=start,
                end_time=end,
                download_before_scan=True,
                repair_gaps=True,
            )

            result = orch.run(request)

            assert result.total_intervals == 2
            assert ("BTCUSDT", "1d") in result.timeframe_results
            assert ("BTCUSDT", "1h") in result.timeframe_results

            for key, tf in result.timeframe_results.items():
                assert tf.download_status == DownloadStatus.COMPLETED
                assert tf.stored_candles > 0

            _integrity_check(db)

    def test_live_gap_repair_roundtrip(self, db):
        """Download → delete candles → repair → verify continuity."""
        with BinanceRestClient() as client:
            syncer = UniverseSyncer(db=db, client=client)
            syncer.sync_spot_universe(quote_asset="USDT")

            downloader = HistoricalKlineDownloader(
                db=db, client=client, request_delay_ms=300.0
            )

            now_ms = int(time.time() * 1000)
            start = now_ms - (90 * DAY)
            end = start + (3 * DAY)

            # Step 1: Initial download
            dl_result = downloader.download_historical_klines(
                symbol="BTCUSDT", interval="1d",
                start_time=start, end_time=end,
            )
            assert dl_result.records_stored > 0

            # Step 2: Verify coverage is complete
            scanner = CoverageScanner(db)
            report = scanner.scan_coverage("BTCUSDT", "1d", start, end,
                                           server_time_ms=client.get_server_time().server_time_ms)
            initial_candles = report.actual_candles

            if initial_candles >= 2:
                # Step 3: Delete a candle to create an interior gap
                open_times = db.query_kline_open_times_range("BTCUSDT", "1d", start, end)
                if len(open_times) >= 2:
                    middle = open_times[len(open_times) // 2]
                    _delete_klines(db, "BTCUSDT", "1d", [middle])

                    # Step 4: Scan again — should find gap
                    report2 = scanner.scan_coverage(
                        "BTCUSDT", "1d", start, end,
                        server_time_ms=client.get_server_time().server_time_ms,
                    )
                    assert report2.is_complete is False
                    assert report2.missing_candles >= 1

                    # Step 5: Repair
                    segments = GapRepairer.plan_repair_segments(report2)
                    assert len(segments) >= 1
                    results = GapRepairer.execute_repairs(segments, downloader)
                    assert all(r.success for r in results)

                    # Step 6: Verify
                    final_report, status = GapRepairer.verify_repairs(
                        "BTCUSDT", "1d", start, end, scanner,
                        server_time_ms=client.get_server_time().server_time_ms,
                    )
                    assert status == RepairStatus.COMPLETED
                    assert final_report.is_complete is True

            _integrity_check(db)
