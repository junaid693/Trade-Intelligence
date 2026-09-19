"""Tests for HistoricalDataOrchestrator (unit-level with mocked downloader)."""

import os
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from trade_intelligence.binance.client import BinanceRestClient
from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.binance.exceptions import BinanceConnectionError
from trade_intelligence.binance.models import ServerTime
from trade_intelligence.db.database import Database
from trade_intelligence.klines.downloader import HistoricalKlineDownloader
from trade_intelligence.klines.gap_detector import INTERVAL_MS
from trade_intelligence.klines.pipeline.orchestrator import HistoricalDataOrchestrator
from trade_intelligence.klines.pipeline.types import (
    CoverageStatus,
    DownloadStatus,
    PipelineConfig,
    PipelineRequest,
    PipelineResult,
    PipelineStatus,
    RepairStatus,
)
from trade_intelligence.klines.types import DownloadResult, GapReport


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
def mock_client():
    client = MagicMock(spec=BinanceRestClient)
    # Default server time far in the future so no forming-candle clamping
    client.get_server_time.return_value = ServerTime.from_raw(
        {"serverTime": 999_999_999_999_999}
    )
    return client


@pytest.fixture
def mock_downloader():
    return MagicMock(spec=HistoricalKlineDownloader)


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
    interval_ms = INTERVAL_MS.get(interval, HOUR)
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


def _make_download_result(symbol="BTCUSDT", interval="1h", stored=10, start=0, end=9*HOUR):
    return DownloadResult(
        symbol=symbol, interval=interval, run_id=1,
        requested_start_time=start, requested_end_time=end,
        effective_start_time=start, effective_end_time=end,
        pages_fetched=1, records_fetched=stored, records_stored=stored,
        gap_report=GapReport(
            symbol=symbol, interval=interval,
            total_expected_candles=stored, total_actual_candles=stored,
            total_missing_candles=0, coverage_ratio=1.0,
        ),
        execution_time_seconds=0.1, status="completed",
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestOrchestratorConstructor:
    def test_invalid_db_type(self, mock_client, mock_downloader):
        with pytest.raises(TypeError, match="Database"):
            HistoricalDataOrchestrator("not_db", mock_client, mock_downloader)

    def test_invalid_client_type(self, db, mock_downloader):
        with pytest.raises(TypeError, match="BinanceRestClient"):
            HistoricalDataOrchestrator(db, "not_client", mock_downloader)

    def test_invalid_downloader_type(self, db, mock_client):
        with pytest.raises(TypeError, match="HistoricalKlineDownloader"):
            HistoricalDataOrchestrator(db, mock_client, "not_downloader")


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

class TestRequestValidation:
    def test_empty_symbols(self, db, mock_client, mock_downloader):
        orch = HistoricalDataOrchestrator(db, mock_client, mock_downloader)
        request = PipelineRequest(
            symbols=[], intervals=["1h"], start_time=0, end_time=HOUR
        )
        with pytest.raises(ValueError, match="symbols"):
            orch.run(request)

    def test_invalid_interval(self, db, mock_client, mock_downloader):
        orch = HistoricalDataOrchestrator(db, mock_client, mock_downloader)
        request = PipelineRequest(
            symbols=["BTCUSDT"], intervals=["2h"], start_time=0, end_time=HOUR
        )
        with pytest.raises(ValueError, match="Unsupported"):
            orch.run(request)

    def test_bool_start_time(self, db, mock_client, mock_downloader):
        orch = HistoricalDataOrchestrator(db, mock_client, mock_downloader)
        request = PipelineRequest(
            symbols=["BTCUSDT"], intervals=["1h"], start_time=True, end_time=HOUR
        )
        with pytest.raises(TypeError, match="start_time"):
            orch.run(request)

    def test_start_gte_end(self, db, mock_client, mock_downloader):
        orch = HistoricalDataOrchestrator(db, mock_client, mock_downloader)
        request = PipelineRequest(
            symbols=["BTCUSDT"], intervals=["1h"], start_time=HOUR, end_time=HOUR
        )
        with pytest.raises(ValueError, match="start_time.*<.*end_time"):
            orch.run(request)


# ---------------------------------------------------------------------------
# Orchestration execution tests
# ---------------------------------------------------------------------------

class TestOrchestratorExecution:
    def test_complete_no_gaps(self, db, mock_client, mock_downloader):
        """Pre-seeded complete data → no downloads needed if download_before_scan=False."""
        _seed_symbol(db)
        _seed_klines(db, "BTCUSDT", "1h", [i * HOUR for i in range(10)])

        orch = HistoricalDataOrchestrator(db, mock_client, mock_downloader)
        request = PipelineRequest(
            symbols=["BTCUSDT"], intervals=["1h"],
            start_time=0, end_time=9 * HOUR,
            download_before_scan=False, repair_gaps=False,
        )

        result = orch.run(request)

        assert result.overall_status == PipelineStatus.COMPLETED
        assert result.completed_count == 1
        assert result.failed_count == 0
        tf = result.timeframe_results[("BTCUSDT", "1h")]
        assert tf.coverage_status == CoverageStatus.COMPLETE
        assert tf.repair_status == RepairStatus.NOT_NEEDED

    def test_download_then_scan(self, db, mock_client, mock_downloader):
        """Downloader is called first, then scanner evaluates coverage."""
        _seed_symbol(db)

        def side_effect(**kwargs):
            # Simulate downloader seeding data
            _seed_klines(db, "BTCUSDT", "1h", [i * HOUR for i in range(6)])
            return _make_download_result(stored=6, end=5*HOUR)

        mock_downloader.download_historical_klines.side_effect = side_effect

        orch = HistoricalDataOrchestrator(db, mock_client, mock_downloader)
        request = PipelineRequest(
            symbols=["BTCUSDT"], intervals=["1h"],
            start_time=0, end_time=5 * HOUR,
            download_before_scan=True, repair_gaps=False,
        )

        result = orch.run(request)

        assert result.overall_status == PipelineStatus.COMPLETED
        mock_downloader.download_historical_klines.assert_called_once()
        tf = result.timeframe_results[("BTCUSDT", "1h")]
        assert tf.download_status == DownloadStatus.COMPLETED
        assert tf.coverage_status == CoverageStatus.COMPLETE

    def test_download_failure_isolated(self, db, mock_client, mock_downloader):
        """Download failure for one pair doesn't halt the other."""
        _seed_symbol(db, "BTCUSDT")
        _seed_symbol(db, "ETHUSDT")
        _seed_klines(db, "ETHUSDT", "1h", [i * HOUR for i in range(6)])

        mock_downloader.download_historical_klines.side_effect = [
            BinanceConnectionError("timeout"),    # BTCUSDT fails
            _make_download_result("ETHUSDT", stored=6, end=5*HOUR),  # ETHUSDT ok
        ]

        orch = HistoricalDataOrchestrator(db, mock_client, mock_downloader)
        request = PipelineRequest(
            symbols=["BTCUSDT", "ETHUSDT"], intervals=["1h"],
            start_time=0, end_time=5 * HOUR,
            download_before_scan=True, repair_gaps=False,
        )

        result = orch.run(request)

        assert result.overall_status == PipelineStatus.PARTIAL_FAILURE
        assert result.failed_count == 1
        assert result.completed_count == 1

        btc = result.timeframe_results[("BTCUSDT", "1h")]
        assert btc.download_status == DownloadStatus.FAILED
        assert btc.error_message is not None

        eth = result.timeframe_results[("ETHUSDT", "1h")]
        assert eth.download_status == DownloadStatus.COMPLETED


class TestFailFast:
    def test_fail_fast_aborts_on_first_failure(self, db, mock_client, mock_downloader):
        _seed_symbol(db, "BTCUSDT")
        _seed_symbol(db, "ETHUSDT")

        mock_downloader.download_historical_klines.side_effect = BinanceConnectionError("boom")

        orch = HistoricalDataOrchestrator(db, mock_client, mock_downloader)
        request = PipelineRequest(
            symbols=["BTCUSDT", "ETHUSDT"], intervals=["1h"],
            start_time=0, end_time=5 * HOUR,
            download_before_scan=True, repair_gaps=False,
            config=PipelineConfig(fail_fast=True),
        )

        result = orch.run(request)

        # Only BTCUSDT was attempted
        assert ("BTCUSDT", "1h") in result.timeframe_results
        assert ("ETHUSDT", "1h") not in result.timeframe_results
        assert result.overall_status == PipelineStatus.FAILED


class TestMultipleIntervals:
    def test_symbol_first_order(self, db, mock_client, mock_downloader):
        """Verify sequential symbol-first, then interval ordering."""
        _seed_symbol(db, "BTCUSDT")
        _seed_klines(db, "BTCUSDT", "1h", [i * HOUR for i in range(6)])
        _seed_klines(db, "BTCUSDT", "1d", [i * DAY for i in range(2)])

        orch = HistoricalDataOrchestrator(db, mock_client, mock_downloader)
        request = PipelineRequest(
            symbols=["BTCUSDT"], intervals=["1h", "1d"],
            start_time=0, end_time=5 * HOUR,
            download_before_scan=False, repair_gaps=False,
        )

        result = orch.run(request)

        assert ("BTCUSDT", "1h") in result.timeframe_results
        assert ("BTCUSDT", "1d") in result.timeframe_results
        assert result.total_intervals == 2


class TestThreeAxisStatus:
    def test_downloaded_but_gaps_found(self, db, mock_client, mock_downloader):
        """download_status=COMPLETED can coexist with coverage_status=GAPS_FOUND."""
        _seed_symbol(db)

        def side_effect(**kwargs):
            # Seed incomplete data
            _seed_klines(db, "BTCUSDT", "1h", [0, HOUR, 4 * HOUR, 5 * HOUR])
            return _make_download_result(stored=4, end=5*HOUR)

        mock_downloader.download_historical_klines.side_effect = side_effect

        orch = HistoricalDataOrchestrator(db, mock_client, mock_downloader)
        request = PipelineRequest(
            symbols=["BTCUSDT"], intervals=["1h"],
            start_time=0, end_time=5 * HOUR,
            download_before_scan=True, repair_gaps=False,
        )

        result = orch.run(request)

        tf = result.timeframe_results[("BTCUSDT", "1h")]
        assert tf.download_status == DownloadStatus.COMPLETED
        assert tf.coverage_status == CoverageStatus.GAPS_FOUND
        assert tf.repair_status == RepairStatus.NOT_REQUESTED


class TestPipelineResult:
    def test_aggregate_counts(self, db, mock_client, mock_downloader):
        _seed_symbol(db, "BTCUSDT")
        _seed_symbol(db, "ETHUSDT")
        _seed_klines(db, "BTCUSDT", "1h", [i * HOUR for i in range(6)])
        _seed_klines(db, "ETHUSDT", "1h", [i * HOUR for i in range(6)])

        orch = HistoricalDataOrchestrator(db, mock_client, mock_downloader)
        request = PipelineRequest(
            symbols=["BTCUSDT", "ETHUSDT"], intervals=["1h"],
            start_time=0, end_time=5 * HOUR,
            download_before_scan=False, repair_gaps=False,
        )

        result = orch.run(request)

        assert result.total_symbols == 2
        assert result.total_intervals == 1
        assert result.completed_count == 2
        assert result.failed_count == 0
        assert result.execution_time_seconds >= 0


class TestGapRepairIntegration:
    def test_repair_flow_end_to_end(self, db, mock_client, mock_downloader):
        """Full flow: download (partial) → scan → plan → repair → verify."""
        _seed_symbol(db)

        # Phase 1: initial download seeds incomplete data
        def initial_download(**kwargs):
            _seed_klines(db, "BTCUSDT", "1h", [0, HOUR, 2 * HOUR, 5 * HOUR])
            return _make_download_result(stored=4, end=5*HOUR)

        # Phase 2: repair download fills the gap
        def repair_download(**kwargs):
            _seed_klines(db, "BTCUSDT", "1h", [3 * HOUR, 4 * HOUR])
            return _make_download_result(stored=2, start=3*HOUR, end=4*HOUR)

        # We need to call the functions as the side effect runs
        call_count = [0]
        def dl_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return initial_download(**kwargs)
            else:
                return repair_download(**kwargs)

        mock_downloader.download_historical_klines.side_effect = dl_side_effect

        orch = HistoricalDataOrchestrator(db, mock_client, mock_downloader)
        request = PipelineRequest(
            symbols=["BTCUSDT"], intervals=["1h"],
            start_time=0, end_time=5 * HOUR,
            download_before_scan=True, repair_gaps=True,
        )

        result = orch.run(request)

        tf = result.timeframe_results[("BTCUSDT", "1h")]
        assert tf.download_status == DownloadStatus.COMPLETED
        assert tf.coverage_status == CoverageStatus.COMPLETE
        assert tf.repair_status == RepairStatus.COMPLETED
        assert result.total_repairs_attempted >= 1
        assert result.total_repairs_completed >= 1
