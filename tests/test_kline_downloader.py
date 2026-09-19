"""Comprehensive unit and offline tests for HistoricalKlineDownloader (Phase 1.4.2).

Verifies all documented acceptance gates:
1. Pagination cannot stall.
2. Forming candles never enter historical storage.
3. All-forming pages terminate safely.
4. close_time + 1 pagination boundaries are correct.
5. Tail Resume really resumes from the latest persisted candle.
6. Tail Resume does not pretend to detect old interior gaps.
7. Raw response provenance links correctly to every stored page.
8. Raw persistence and normalized persistence failure behavior matches documented transaction boundaries.
9. Retry/429 behavior actually works rather than merely existing in comments.
10. Idempotency preserves row counts.
11. Delisted/relisted symbols respect existing FK behavior.
12. SQLite integrity and foreign-key checks remain clean.
"""

from decimal import Decimal
import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from trade_intelligence.binance.client import BinanceRestClient
from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.binance.exceptions import (
    BinanceConnectionError,
    BinanceHttpError,
    BinanceTimeoutError,
)
from trade_intelligence.binance.models import Kline, ServerTime
from trade_intelligence.db.database import Database
from trade_intelligence.db.exceptions import DatabaseError
from trade_intelligence.klines.downloader import HistoricalKlineDownloader
from trade_intelligence.klines.gap_detector import (
    detect_kline_gaps,
    interval_to_milliseconds,
)
from trade_intelligence.klines.types import DownloadResult, GapReport, KlineGap


def _make_mock_kline(
    open_time_ms: int,
    interval_ms: int = 3600000,
    open_price: str = "50000.00",
    close_price: str = "50100.00",
    volume: str = "10.5",
    trades: int = 150,
) -> Kline:
    """Helper to build an authentic Kline dataclass instance."""
    close_time_ms = open_time_ms + interval_ms - 1
    raw = [
        open_time_ms,
        open_price,
        "50500.00",
        "49500.00",
        close_price,
        volume,
        close_time_ms,
        "526050.00",
        trades,
        "5.25",
        "263025.00",
        "0",
    ]
    return Kline.from_raw(raw)


class TestKlineDownloader(unittest.TestCase):
    """Offline test suite for HistoricalKlineDownloader."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_downloader.db")
        self.db = Database(self.db_path)
        self.db.connect()
        self.db.initialize()

        # Seed BTCUSDT symbol
        self.db.insert_symbol(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            status="TRADING",
            is_spot_trading_allowed=True,
            is_margin_trading_allowed=True,
            base_asset_precision=8,
            quote_asset_precision=8,
            updated_at=1700000000000,
        )

        self.mock_client = MagicMock(spec=BinanceRestClient)
        # Mock session for header capturing
        self.mock_session = MagicMock()
        self.mock_session.hooks = {"response": []}
        self.mock_client._session = self.mock_session

        # Default server time: far in future so standard candles are closed
        self.server_time_ms = 1800000000000
        self.mock_client.get_server_time.return_value = ServerTime.from_raw({"serverTime": self.server_time_ms})

        self.downloader = HistoricalKlineDownloader(
            db=self.db,
            client=self.mock_client,
            request_delay_ms=0.0,  # Fast offline tests
            max_retries=3,
        )

    def tearDown(self):
        self.db.close()

    def test_download_single_page_success(self):
        """Gate 4 & 7: Verify single-page download, normalization, and provenance."""
        base_time = 1700000000000
        mock_klines = [_make_mock_kline(base_time + i * 3600000) for i in range(500)]
        self.mock_client.get_klines.return_value = mock_klines

        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 500 * 3600000,
            resume=False,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.records_fetched, 500)
        self.assertEqual(result.records_stored, 500)
        self.assertEqual(result.pages_fetched, 1)

        # Verify database contents
        rows = self.db.query_klines_range("BTCUSDT", "1h", base_time, base_time + 600 * 3600000)
        self.assertEqual(len(rows), 500)

        # Verify provenance
        conn = self.db.connection
        raw_rows = conn.execute("SELECT * FROM raw_api_responses;").fetchall()
        self.assertEqual(len(raw_rows), 1)
        raw_resp = raw_rows[0]
        self.assertEqual(raw_resp["endpoint"], "/api/v3/klines")
        self.assertEqual(raw_resp["ingestion_run_id"], result.run_id)

        # Confirm all klines reference this raw response
        for r in rows:
            self.assertEqual(r["raw_response_id"], raw_resp["id"])

        # Check ingestion run
        run_row = conn.execute("SELECT * FROM ingestion_runs WHERE id = ?;", (result.run_id,)).fetchone()
        self.assertEqual(run_row["status"], "completed")
        self.assertEqual(run_row["records_stored"], 500)

    def test_pagination_close_time_plus_one_boundaries(self):
        """Gate 4: Verify pagination advances by close_time + 1 across multiple pages."""
        base_time = 1700000000000 - (1700000000000 % 3600000)
        # Page 1: 1000 candles
        page1 = [_make_mock_kline(base_time + i * 3600000) for i in range(1000)]
        # Page 2: 500 candles
        page2_start = page1[-1].close_time_ms + 1
        page2 = [_make_mock_kline(page2_start + i * 3600000) for i in range(500)]

        # On third call return empty list (exhausted)
        self.mock_client.get_klines.side_effect = [page1, page2, []]

        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 2000 * 3600000,
            resume=False,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.records_stored, 1500)
        self.assertEqual(result.pages_fetched, 2)

        # Verify call arguments
        calls = self.mock_client.get_klines.call_args_list
        self.assertEqual(len(calls), 2)
        # First call started at base_time
        self.assertEqual(calls[0].kwargs["start_time"], base_time)
        # Second call started at page1[-1].close_time_ms + 1
        self.assertEqual(calls[1].kwargs["start_time"], page2_start)

    def test_pagination_cannot_stall_raises_runtime_error(self):
        """Gate 1: If next_start_time does not advance, raise RuntimeError immediately."""
        base_time = 1700000000000 - (1700000000000 % 3600000)
        page1 = [_make_mock_kline(base_time + i * 3600000) for i in range(1000)]
        # API returns identical 1000 candles on second page instead of advancing
        self.mock_client.get_klines.side_effect = [page1, page1]

        with self.assertRaises(RuntimeError) as ctx:
            self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=base_time,
                end_time=base_time + 5000 * 3600000,
                resume=False,
            )

        self.assertIn("Pagination stalled", str(ctx.exception))

        # Ingestion run should be marked failed
        conn = self.db.connection
        run = conn.execute("SELECT * FROM ingestion_runs ORDER BY id DESC LIMIT 1;").fetchone()
        self.assertEqual(run["status"], "failed")
        self.assertIn("Pagination stalled", run["error_message"])

    def test_forming_candles_never_enter_historical_storage(self):
        """Gate 2: Candles with close_time >= server_time are discarded."""
        server_time = 1700000000000 + 4 * 3600000
        self.mock_client.get_server_time.return_value = ServerTime.from_raw({"serverTime": server_time})

        # Create 5 candles: candles 0, 1, 2, 3 are closed; candle 4 closes at server_time + 3599999
        base_time = 1700000000000
        klines = [_make_mock_kline(base_time + i * 3600000) for i in range(5)]
        self.mock_client.get_klines.return_value = klines

        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 10 * 3600000,
            resume=False,
        )

        # Only closed candles should be stored
        self.assertEqual(result.records_stored, 4)
        stored = self.db.query_klines_range("BTCUSDT", "1h", base_time, base_time + 10 * 3600000)
        self.assertEqual(len(stored), 4)
        for s in stored:
            self.assertLess(s["close_time"], server_time)

    def test_all_forming_pages_terminate_safely(self):
        """Gate 3: If all candles in batch are forming, terminate safely without advance from discarded."""
        server_time = 1700000000000
        self.mock_client.get_server_time.return_value = ServerTime.from_raw({"serverTime": server_time})

        # Every candle is forming (open_time >= server_time)
        forming_klines = [_make_mock_kline(server_time + i * 3600000) for i in range(3)]
        self.mock_client.get_klines.return_value = forming_klines

        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=server_time,
            end_time=server_time + 10 * 3600000,
            resume=False,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.records_stored, 0)
        # Should have stopped after 1 batch without getting stuck in a loop
        self.assertEqual(result.pages_fetched, 1)

    def test_tail_resume_resumes_from_latest_persisted_candle(self):
        """Gate 5: Tail Resume queries latest candle and starts at latest.close_time + 1."""
        base_time = 1700000000000
        # Pre-seed 100 candles in DB
        pre_seeded = [_make_mock_kline(base_time + i * 3600000) for i in range(100)]
        self.mock_client.get_klines.return_value = pre_seeded
        self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 100 * 3600000,
            resume=False,
        )
        self.mock_client.get_klines.reset_mock()

        # Second download with resume=True
        next_start = pre_seeded[-1].close_time_ms + 1
        new_batch = [_make_mock_kline(next_start + i * 3600000) for i in range(50)]
        self.mock_client.get_klines.return_value = new_batch

        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 200 * 3600000,
            resume=True,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.records_stored, 50)
        self.assertEqual(result.effective_start_time, next_start)
        # Verify client was called starting at next_start
        self.mock_client.get_klines.assert_called_once()
        self.assertEqual(self.mock_client.get_klines.call_args.kwargs["start_time"], next_start)

    def test_tail_resume_already_up_to_date(self):
        """Gate 5: If tail is already at or past effective_end_time, report already_up_to_date."""
        base_time = 1700000000000
        pre_seeded = [_make_mock_kline(base_time + i * 3600000) for i in range(10)]
        self.mock_client.get_klines.return_value = pre_seeded
        self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 9 * 3600000,
            resume=False,
        )
        self.mock_client.get_klines.reset_mock()

        # Resume over same covered range
        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 9 * 3600000,
            resume=True,
        )

        self.assertEqual(result.status, "already_up_to_date")
        self.assertEqual(result.records_stored, 0)
        self.mock_client.get_klines.assert_not_called()

    def test_tail_resume_does_not_pretend_to_detect_old_interior_gaps(self):
        """Gate 6: Tail Resume gap report describes only the examined sequence."""
        base_time = 1700000000000
        # Pre-seed DB with candle 0 and candle 2 (interior gap at candle 1)
        k0 = _make_mock_kline(base_time)
        k2 = _make_mock_kline(base_time + 2 * 3600000)
        self.mock_client.get_klines.return_value = [k0, k2]
        self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 3 * 3600000,
            resume=False,
        )
        self.mock_client.get_klines.reset_mock()

        # Tail Resume: starts at k2.close_time_ms + 1 = base_time + 3h
        k3 = _make_mock_kline(base_time + 3 * 3600000)
        k4 = _make_mock_kline(base_time + 4 * 3600000)
        self.mock_client.get_klines.return_value = [k3, k4]

        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 4 * 3600000,
            resume=True,
        )

        # Tail resume starts at k3 (base_time + 3h). Examined sequence [k3, k4] has 0 gaps.
        self.assertEqual(result.gap_report.total_actual_candles, 2)
        self.assertEqual(result.gap_report.total_missing_candles, 0)
        self.assertEqual(len(result.gap_report.gaps), 0)

    def test_gap_detection_identifies_missing_intervals(self):
        """Gate 6: GapReport correctly identifies missing intervals in examined sequence."""
        base_time = 1700000000000
        # Sequence with 2-hour gap between 1h and 4h (missing 2h and 3h)
        k0 = _make_mock_kline(base_time)
        k1 = _make_mock_kline(base_time + 1 * 3600000)
        k4 = _make_mock_kline(base_time + 4 * 3600000)
        k5 = _make_mock_kline(base_time + 5 * 3600000)
        self.mock_client.get_klines.return_value = [k0, k1, k4, k5]

        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 5 * 3600000,
            resume=False,
        )

        # Completion status is still 'completed' (API work succeeded)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.gap_report.total_actual_candles, 4)
        self.assertEqual(result.gap_report.total_missing_candles, 2)
        self.assertEqual(len(result.gap_report.gaps), 1)

        gap = result.gap_report.gaps[0]
        self.assertEqual(gap.missing_candles, 2)
        self.assertEqual(gap.gap_start_ms, base_time + 2 * 3600000)
        self.assertEqual(gap.gap_end_ms, base_time + 4 * 3600000 - 1)

    def test_raw_persistence_and_normalized_transaction_boundary(self):
        """Gate 8: If batch_upsert fails, raw response remains committed for audit."""
        base_time = 1700000000000
        self.mock_client.get_klines.return_value = [_make_mock_kline(base_time)]

        # Simulate failure in batch_upsert_klines
        with patch.object(self.db, "batch_upsert_klines", side_effect=DatabaseError("Disk full")):
            with self.assertRaises(DatabaseError):
                self.downloader.download_historical_klines(
                    symbol="BTCUSDT",
                    interval="1h",
                    start_time=base_time,
                    end_time=base_time + 3600000,
                    resume=False,
                )

        conn = self.db.connection
        # Raw response WAS committed
        raw_rows = conn.execute("SELECT * FROM raw_api_responses;").fetchall()
        self.assertEqual(len(raw_rows), 1)

        # Ingestion run was marked failed
        run = conn.execute("SELECT * FROM ingestion_runs ORDER BY id DESC LIMIT 1;").fetchone()
        self.assertEqual(run["status"], "failed")
        self.assertIn("Disk full", run["error_message"])

    def test_retry_and_429_handling(self):
        """Gate 9: HTTP 429 parses Retry-After and retries successfully."""
        base_time = 1700000000000
        klines = [_make_mock_kline(base_time)]

        # First call raises 429, second call succeeds
        http_429 = BinanceHttpError(status_code=429, message="Rate limit exceeded")
        self.mock_client.get_klines.side_effect = [http_429, klines]

        # Simulate response header Retry-After: 1
        self.downloader._last_headers = {"retry-after": "1"}

        with patch("time.sleep") as mock_sleep:
            result = self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=base_time,
                end_time=base_time + 3600000,
                resume=False,
            )
            # Sleep was called with 1.0s
            mock_sleep.assert_any_call(1.0)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.records_stored, 1)

    def test_retry_transient_timeout_error(self):
        """Gate 9: Transient timeout retries with exponential backoff."""
        base_time = 1700000000000
        klines = [_make_mock_kline(base_time)]

        timeout_err = BinanceTimeoutError("Request timed out")
        self.mock_client.get_klines.side_effect = [timeout_err, klines]

        with patch("time.sleep") as mock_sleep:
            result = self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=base_time,
                end_time=base_time + 3600000,
                resume=False,
            )
            mock_sleep.assert_any_call(1.0)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.records_stored, 1)

    def test_idempotency_preserves_row_counts(self):
        """Gate 10: Repeated downloads over the same range keep row count invariant."""
        base_time = 1700000000000
        klines = [_make_mock_kline(base_time + i * 3600000) for i in range(20)]
        self.mock_client.get_klines.return_value = klines

        # First download
        res1 = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 20 * 3600000,
            resume=False,
        )
        self.assertEqual(res1.records_stored, 20)

        conn = self.db.connection
        cnt1 = conn.execute("SELECT COUNT(*) AS c FROM klines WHERE symbol = 'BTCUSDT';").fetchone()["c"]
        self.assertEqual(cnt1, 20)

        # Second download (force re-download with resume=False)
        res2 = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 20 * 3600000,
            resume=False,
        )
        self.assertEqual(res2.records_stored, 20)

        cnt2 = conn.execute("SELECT COUNT(*) AS c FROM klines WHERE symbol = 'BTCUSDT';").fetchone()["c"]
        self.assertEqual(cnt2, 20, "Row count must remain invariant under repeated downloads")

    def test_delisted_symbol_foreign_key_protection(self):
        """Gate 11: Nonexistent symbol rejected; delisted symbol succeeds."""
        base_time = 1700000000000
        # 1. Nonexistent symbol raises ValueError before any API call
        with self.assertRaises(ValueError) as ctx:
            self.downloader.download_historical_klines(
                symbol="UNKNOWNUSDT",
                interval="1h",
                start_time=base_time,
                end_time=base_time + 3600000,
            )
        self.assertIn("not found in symbols table", str(ctx.exception))
        self.mock_client.get_klines.assert_not_called()

        # 2. Delisted symbol in symbols table succeeds
        self.db.insert_symbol(
            symbol="LUNAUSDT",
            base_asset="LUNA",
            quote_asset="USDT",
            status="DELISTED",
            is_spot_trading_allowed=False,
            is_margin_trading_allowed=False,
            base_asset_precision=8,
            quote_asset_precision=8,
            updated_at=1700000000000,
        )

        self.mock_client.get_klines.return_value = [
            _make_mock_kline(base_time, open_price="1.00", close_price="0.50")
        ]
        result = self.downloader.download_historical_klines(
            symbol="LUNAUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 3600000,
            resume=False,
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.records_stored, 1)

    def test_sqlite_integrity_and_foreign_keys(self):
        """Gate 14: Confirm PRAGMA integrity_check and foreign_key_check are clean."""
        base_time = 1700000000000
        mock_klines = [_make_mock_kline(base_time + i * 3600000) for i in range(10)]
        self.mock_client.get_klines.return_value = mock_klines

        self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 10 * 3600000,
            resume=False,
        )

        conn = self.db.connection
        fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
        self.assertEqual(len(fk_violations), 0, f"Foreign key violations found: {fk_violations}")

        integrity = conn.execute("PRAGMA integrity_check;").fetchone()
        self.assertEqual(integrity[0], "ok")

    def test_pagination_exact_1000_candles_no_extra_request(self):
        """Area 1: Exact 1000 candles completes in 1 page without extra empty request."""
        base_time = 1700000000000
        page = [_make_mock_kline(base_time + i * 3600000) for i in range(1000)]
        self.mock_client.get_klines.return_value = page

        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 999 * 3600000,
            resume=False,
        )

        self.assertEqual(result.records_stored, 1000)
        self.assertEqual(result.pages_fetched, 1)
        self.mock_client.get_klines.assert_called_once()

    def test_pagination_unordered_candles_sorted_safely(self):
        """Area 1 & Bug 2: Out-of-order candles in API page are sorted before advancement."""
        base_time = 1700000000000
        k0 = _make_mock_kline(base_time)
        k1 = _make_mock_kline(base_time + 3600000)
        k2 = _make_mock_kline(base_time + 2 * 3600000)
        # API returns candles in scrambled order: [k2, k0, k1]
        self.mock_client.get_klines.return_value = [k2, k0, k1]

        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 2 * 3600000,
            resume=False,
        )

        self.assertEqual(result.records_stored, 3)
        # Verify rows in DB are ordered and next start time was advanced past k2
        rows = self.db.query_klines_range("BTCUSDT", "1h", base_time, base_time + 3 * 3600000)
        self.assertEqual([r["open_time"] for r in rows], [base_time, base_time + 3600000, base_time + 2 * 3600000])

    def test_pagination_empty_api_response_on_historical_range(self):
        """Area 1 & 4 (Bug 4): Empty response for historical past range reports accurate gap."""
        base_time = 1700000000000
        self.mock_client.get_klines.return_value = []

        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 4 * 3600000,
            resume=False,
        )

        self.assertEqual(result.records_stored, 0)
        self.assertEqual(result.pages_fetched, 1)
        self.assertEqual(result.gap_report.total_expected_candles, 5)
        self.assertEqual(result.gap_report.total_actual_candles, 0)
        self.assertEqual(result.gap_report.total_missing_candles, 5)
        self.assertEqual(result.gap_report.coverage_ratio, 0.0)

    def test_forming_candle_server_time_equals_close_time_excluded(self):
        """Area 2: Candle with close_time equal to server_time is treated as forming."""
        server_time = 1700000000000 + 3600000 - 1
        self.mock_client.get_server_time.return_value = ServerTime.from_raw({"serverTime": server_time})

        # Candle 0 closes exactly at server_time
        k0 = _make_mock_kline(1700000000000)
        self.assertEqual(k0.close_time_ms, server_time)
        self.mock_client.get_klines.return_value = [k0]

        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=1700000000000,
            end_time=1700000000000 + 3600000,
            resume=False,
        )
        self.assertEqual(result.records_stored, 0, "Candle closing at server_time must be excluded")

    def test_tail_resume_with_existing_candle_outside_range(self):
        """Area 3: Tail resume ignores existing candles outside the requested bounds."""
        base_time = 1700000000000 - (1700000000000 % 3600000)
        # Seed an older candle outside requested start_time
        older_candle = _make_mock_kline(base_time - 100 * 3600000)
        self.mock_client.get_klines.return_value = [older_candle]
        self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time - 100 * 3600000,
            end_time=base_time - 99 * 3600000,
            resume=False,
        )
        self.mock_client.get_klines.reset_mock()

        # Download a new window starting at base_time with resume=True
        new_candle = _make_mock_kline(base_time)
        self.mock_client.get_klines.return_value = [new_candle]
        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 3600000,
            resume=True,
        )

        # Must start from base_time, ignoring older_candle
        self.assertEqual(result.effective_start_time, base_time)
        self.assertEqual(result.records_stored, 1)

    def test_tail_resume_after_partial_page_failure(self):
        """Area 3 & 7: Successful resume after partial failure with clean provenance."""
        base_time = 1700000000000 - (1700000000000 % 3600000)
        page1 = [_make_mock_kline(base_time + i * 3600000) for i in range(1000)]
        # Page 1 succeeds, Page 2 raises transient connection error
        self.mock_client.get_klines.side_effect = [
            page1,
            BinanceConnectionError("Connection lost on page 2"),
            BinanceConnectionError("Connection lost on page 2"),
            BinanceConnectionError("Connection lost on page 2"),
            BinanceConnectionError("Connection lost on page 2"),
        ]

        with self.assertRaises(BinanceConnectionError):
            self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=base_time,
                end_time=base_time + 1500 * 3600000,
                resume=False,
            )

        # Page 1 survived in DB
        rows_page1 = self.db.query_klines_range("BTCUSDT", "1h", base_time, base_time + 1500 * 3600000)
        self.assertEqual(len(rows_page1), 1000)

        # Now resume
        page2_start = page1[-1].close_time_ms + 1
        page2 = [_make_mock_kline(page2_start + i * 3600000) for i in range(500)]
        self.mock_client.get_klines.side_effect = [page2]

        resume_result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 1500 * 3600000,
            resume=True,
        )

        self.assertEqual(resume_result.status, "completed")
        self.assertEqual(resume_result.records_stored, 500)
        total_rows = self.db.query_klines_range("BTCUSDT", "1h", base_time, base_time + 2000 * 3600000)
        self.assertEqual(len(total_rows), 1500)

    def test_idempotency_overlapping_ranges(self):
        """Area 5: Repeated download with overlapping range updates without row duplication."""
        base_time = 1700000000000
        # First download: hours 0..10 (11 candles)
        batch1 = [_make_mock_kline(base_time + i * 3600000) for i in range(11)]
        self.mock_client.get_klines.return_value = batch1
        self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 10 * 3600000,
            resume=False,
        )

        # Second download: hours 5..15 (11 candles, hours 5..10 overlap)
        batch2 = [_make_mock_kline(base_time + (5 + i) * 3600000) for i in range(11)]
        self.mock_client.get_klines.return_value = batch2
        self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time + 5 * 3600000,
            end_time=base_time + 15 * 3600000,
            resume=False,
        )

        # Total distinct candles must be exactly 16 (0 to 15)
        total_rows = self.db.query_klines_range("BTCUSDT", "1h", base_time, base_time + 20 * 3600000)
        self.assertEqual(len(total_rows), 16)

    def test_provenance_canonical_reconstruction_identical_to_binance(self):
        """Area 6: Stored JSON payload in raw_api_responses strictly matches Binance raw items."""
        base_time = 1700000000000
        mock_kline = _make_mock_kline(base_time)
        self.mock_client.get_klines.return_value = [mock_kline]

        result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 3600000,
            resume=False,
        )

        conn = self.db.connection
        raw_row = conn.execute(
            "SELECT response_json FROM raw_api_responses WHERE ingestion_run_id = ?;",
            (result.run_id,),
        ).fetchone()

        payload = json.loads(raw_row["response_json"])
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0], mock_kline.raw)

    def test_http_418_ip_ban_not_retried(self):
        """Area 8: HTTP 418 (IP ban) raises immediately without retry loops."""
        self.mock_client.get_klines.side_effect = BinanceHttpError(
            status_code=418,
            message="IP banned until timestamp",
        )

        with self.assertRaises(BinanceHttpError) as ctx:
            self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=1700000000000,
                end_time=1700000000000 + 3600000,
                resume=False,
            )

        self.assertEqual(ctx.exception.status_code, 418)
        self.mock_client.get_klines.assert_called_once()

    def test_http_400_bad_request_not_retried(self):
        """Area 8: HTTP 400 raises immediately without retry loops."""
        self.mock_client.get_klines.side_effect = BinanceHttpError(
            status_code=400,
            message="Illegal characters found in parameter",
        )

        with self.assertRaises(BinanceHttpError) as ctx:
            self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=1700000000000,
                end_time=1700000000000 + 3600000,
                resume=False,
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.mock_client.get_klines.assert_called_once()

    def test_rate_limit_telemetry_high_used_weight_throttles(self):
        """Area 9: X-MBX-USED-WEIGHT > 1000 throttles pacing delay safely."""
        self.downloader._last_headers = {"x-mbx-used-weight-1m": "1050"}
        with patch("time.sleep") as mock_sleep:
            self.downloader._handle_pacing_and_telemetry(base_delay_ms=200.0)
            mock_sleep.assert_called_once_with(1.0)

    def test_rate_limit_telemetry_malformed_header_ignored(self):
        """Area 9: Malformed header value does not crash telemetry parsing."""
        self.downloader._last_headers = {"x-mbx-used-weight-1m": "not-an-int"}
        with patch("time.sleep") as mock_sleep:
            self.downloader._handle_pacing_and_telemetry(base_delay_ms=200.0)
            mock_sleep.assert_called_once_with(0.2)

    def test_symbol_normalization_and_validation(self):
        """Area 10 & 11: Symbol string normalization and type checks."""
        base_time = 1700000000000
        # Lowercase symbol is normalized to uppercase and succeeds
        self.mock_client.get_klines.return_value = [_make_mock_kline(base_time)]
        result = self.downloader.download_historical_klines(
            symbol="btcusdt",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 3600000,
            resume=False,
        )
        self.assertEqual(result.symbol, "BTCUSDT")

        # Non-string symbol raises TypeError
        with self.assertRaises(TypeError):
            self.downloader.download_historical_klines(
                symbol=123,  # type: ignore
                interval="1h",
                start_time=base_time,
                end_time=base_time + 3600000,
            )

        # Empty symbol raises ValueError
        with self.assertRaises(ValueError):
            self.downloader.download_historical_klines(
                symbol="   ",
                interval="1h",
                start_time=base_time,
                end_time=base_time + 3600000,
            )

    def test_validation_rejects_bool_and_invalid_timestamps(self):
        """Area 11 (Bug 1): Timestamps reject bool and non-integer types."""
        # Bool start_time raises TypeError
        with self.assertRaises(TypeError):
            self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=True,  # type: ignore
                end_time=1700000000000,
            )

        # Bool end_time raises TypeError
        with self.assertRaises(TypeError):
            self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=1700000000000,
                end_time=False,  # type: ignore
            )

        # Negative start_time raises ValueError
        with self.assertRaises(ValueError):
            self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=-1,
                end_time=1700000000000,
            )

        # start_time > end_time raises ValueError
        with self.assertRaises(ValueError):
            self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=1700000000000,
                end_time=1600000000000,
            )

    def test_validation_rejects_non_bool_resume(self):
        """Area 11 (Bug 1): resume flag rejects non-boolean types."""
        with self.assertRaises(TypeError):
            self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=1700000000000,
                end_time=1700000000000 + 3600000,
                resume="yes",  # type: ignore
            )

    def test_validation_rejects_invalid_delay_and_retries(self):
        """Area 11 (Bug 1): Downloader constructor validates delay and retries."""
        with self.assertRaises(TypeError):
            HistoricalKlineDownloader(self.db, self.mock_client, request_delay_ms="fast")  # type: ignore
        with self.assertRaises(TypeError):
            HistoricalKlineDownloader(self.db, self.mock_client, request_delay_ms=True)  # type: ignore
        with self.assertRaises(ValueError):
            HistoricalKlineDownloader(self.db, self.mock_client, request_delay_ms=-1.0)
        with self.assertRaises(TypeError):
            HistoricalKlineDownloader(self.db, self.mock_client, max_retries="three")  # type: ignore
        with self.assertRaises(TypeError):
            HistoricalKlineDownloader(self.db, self.mock_client, max_retries=True)  # type: ignore
        with self.assertRaises(ValueError):
            HistoricalKlineDownloader(self.db, self.mock_client, max_retries=-1)


class TestGapDetector(unittest.TestCase):
    """Unit tests for standalone detect_kline_gaps and interval_to_milliseconds."""

    def test_interval_to_milliseconds(self):
        self.assertEqual(interval_to_milliseconds("5m"), 300000)
        self.assertEqual(interval_to_milliseconds("15m"), 900000)
        self.assertEqual(interval_to_milliseconds("1h"), 3600000)
        self.assertEqual(interval_to_milliseconds("4h"), 14400000)
        self.assertEqual(interval_to_milliseconds("1d"), 86400000)
        with self.assertRaises(ValueError):
            interval_to_milliseconds("1w")

    def test_detect_kline_gaps_contiguous(self):
        klines = [{"open_time": 1000 + i * 300000} for i in range(10)]
        report = detect_kline_gaps("BTCUSDT", "5m", klines)
        self.assertEqual(report.total_actual_candles, 10)
        self.assertEqual(report.total_missing_candles, 0)
        self.assertEqual(report.coverage_ratio, 1.0)
        self.assertEqual(len(report.gaps), 0)

    def test_detect_kline_gaps_interior_gap(self):
        # Missing index 2 and 3
        times = [1000, 1000 + 300000, 1000 + 4 * 300000]
        klines = [{"open_time": t} for t in times]
        report = detect_kline_gaps("BTCUSDT", "5m", klines)
        self.assertEqual(report.total_actual_candles, 3)
        self.assertEqual(report.total_missing_candles, 2)
        self.assertEqual(len(report.gaps), 1)
        gap = report.gaps[0]
        self.assertEqual(gap.missing_candles, 2)
        self.assertEqual(gap.gap_start_ms, 1000 + 2 * 300000)
        self.assertEqual(gap.gap_end_ms, 1000 + 4 * 300000 - 1)

    def test_detect_kline_gaps_leading_and_trailing(self):
        base = 10000000
        h = 3600000
        klines = [{"open_time": base + 2 * h}, {"open_time": base + 3 * h}]
        report = detect_kline_gaps(
            "BTCUSDT",
            "1h",
            klines,
            expected_start_time=base,
            expected_end_time=base + 5 * h,
        )
        self.assertEqual(report.total_actual_candles, 2)
        self.assertEqual(report.total_missing_candles, 4)
        self.assertEqual(len(report.gaps), 2)
        self.assertEqual(report.gaps[0].missing_candles, 2)
        self.assertEqual(report.gaps[1].missing_candles, 2)

    def test_gap_detector_duplicate_timestamps_deduplicated(self):
        """Area 4 & Bug 3: Duplicate timestamps are deduplicated and don't inflate counts."""
        klines = [
            {"open_time": 1000},
            {"open_time": 1000},
            {"open_time": 1000 + 3600000},
        ]
        report = detect_kline_gaps("BTCUSDT", "1h", klines)
        self.assertEqual(report.total_actual_candles, 2)
        self.assertEqual(report.total_missing_candles, 0)
        self.assertEqual(report.total_expected_candles, 2)
        self.assertEqual(report.coverage_ratio, 1.0)

    def test_gap_detector_unsorted_input_ordered_safely(self):
        """Area 4: Unsorted input is safely ordered before gap detection."""
        klines = [
            {"open_time": 1000 + 2 * 3600000},
            {"open_time": 1000},
            {"open_time": 1000 + 3600000},
        ]
        report = detect_kline_gaps("BTCUSDT", "1h", klines)
        self.assertEqual(report.total_actual_candles, 3)
        self.assertEqual(report.total_missing_candles, 0)
        self.assertEqual(report.coverage_ratio, 1.0)
        self.assertEqual(len(report.gaps), 0)

    def test_gap_detector_input_validation(self):
        """Area 4: Parameter type and range validation in detect_kline_gaps."""
        with self.assertRaises(TypeError):
            detect_kline_gaps(123, "1h", [])  # type: ignore
        with self.assertRaises(ValueError):
            detect_kline_gaps("   ", "1h", [])
        with self.assertRaises(TypeError):
            detect_kline_gaps("BTCUSDT", "1h", [], expected_start_time=True)  # type: ignore
        with self.assertRaises(ValueError):
            detect_kline_gaps("BTCUSDT", "1h", [], expected_start_time=-1)
        with self.assertRaises(TypeError):
            detect_kline_gaps("BTCUSDT", "1h", [], expected_end_time=False)  # type: ignore
        with self.assertRaises(ValueError):
            detect_kline_gaps("BTCUSDT", "1h", [], expected_end_time=-5)
        with self.assertRaises(ValueError):
            detect_kline_gaps("BTCUSDT", "1h", [], expected_start_time=5000, expected_end_time=1000)
        with self.assertRaises(ValueError):
            detect_kline_gaps("BTCUSDT", "1h", [{"open_time": -100}])
