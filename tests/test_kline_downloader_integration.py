"""Live integration test for HistoricalKlineDownloader against Binance Spot API (Phase 1.4.2).

Gated by environment variable RUN_LIVE_TESTS=1.
Executes authentic multi-page historical download for BTCUSDT, verifies raw archival,
database persistence, foreign key compliance, and tail resume behavior.
"""

import os
import tempfile
import time
import unittest

from trade_intelligence.binance.client import BinanceRestClient
from trade_intelligence.db.database import Database
from trade_intelligence.klines.downloader import HistoricalKlineDownloader


@unittest.skipUnless(
    os.getenv("RUN_LIVE_TESTS") == "1",
    "Live tests disabled. Set RUN_LIVE_TESTS=1 to run.",
)
class TestKlineDownloaderLiveIntegration(unittest.TestCase):
    """Live integration test suite for historical kline download."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_live_downloader.db")
        self.db = Database(self.db_path)
        self.db.connect()
        self.db.initialize()

        # Seed BTCUSDT in symbols table to satisfy foreign key constraints
        self.db.insert_symbol(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            status="TRADING",
            is_spot_trading_allowed=True,
            is_margin_trading_allowed=True,
            base_asset_precision=8,
            quote_asset_precision=8,
            updated_at=int(time.time() * 1000),
        )

        self.client = BinanceRestClient()
        self.downloader = HistoricalKlineDownloader(
            db=self.db,
            client=self.client,
            request_delay_ms=250.0,  # Conservative live pacing
            max_retries=3,
        )

    def tearDown(self):
        self.client.close()
        self.db.close()

    def test_live_historical_kline_download_multi_page_and_resume(self):
        """Perform authentic 2-page historical download (1,200 candles) and verify resume."""
        # Query live server time
        server_time = self.client.get_server_time().server_time_ms

        # Historical window safely in the past (14 days ago to 2 days ago), aligned to 1h boundary
        h_ms = 3600 * 1000
        raw_end = server_time - (2 * 24 * h_ms)
        end_time = raw_end - (raw_end % h_ms)
        # 1200 1h candles = 1200 * 3600000 ms (guarantees >= 2 pages since limit=1000)
        start_time = end_time - (1200 * h_ms)

        # 1. First download run
        result1 = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=start_time,
            end_time=end_time,
            resume=False,
        )

        self.assertEqual(result1.status, "completed")
        self.assertGreaterEqual(result1.records_stored, 1000, "Should store at least 1000 candles")
        self.assertGreaterEqual(result1.pages_fetched, 2, "1200 candles should require at least 2 pages")

        # Verify database rows
        stored = self.db.query_klines_range("BTCUSDT", "1h", start_time, end_time + 3600000)
        self.assertEqual(len(stored), result1.records_stored)

        # Confirm zero forming candles entered the database
        for k in stored:
            self.assertLess(k["close_time"], server_time, "Forming candle must not enter storage")

        # Confirm SQLite integrity and foreign keys
        conn = self.db.connection
        fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
        self.assertEqual(len(fk_violations), 0, f"Foreign key violations found: {fk_violations}")

        integrity = conn.execute("PRAGMA integrity_check;").fetchone()
        self.assertEqual(integrity[0], "ok")

        # Confirm raw_api_responses provenance
        raw_rows = conn.execute(
            "SELECT id, endpoint, content_hash, ingestion_run_id FROM raw_api_responses WHERE ingestion_run_id = ?;",
            (result1.run_id,),
        ).fetchall()
        self.assertGreaterEqual(len(raw_rows), 2)
        for r in raw_rows:
            self.assertEqual(r["endpoint"], "/api/v3/klines")
            self.assertEqual(len(r["content_hash"]), 64)

        # 2. Second download run with resume=True over the exact same range
        result2 = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=start_time,
            end_time=end_time,
            resume=True,
        )

        self.assertEqual(result2.status, "already_up_to_date")
        self.assertEqual(result2.records_stored, 0)
        self.assertEqual(result2.pages_fetched, 0)
