"""End-to-end system integration tests across Phase 1.1 to 1.4.2.

Verifies the complete pipeline on a clean SQLite database with zero shortcuts:
Binance API / Client (1.2)
    ↓
Spot Universe Syncer (1.4.1)
    ↓
symbols & symbol_status_events
    ↓
SQLite Foundation (1.3.1 + 1.3.2) & Batch Persistence (1.3.3)
    ↓
Historical Kline Downloader (1.4.2)
    ↓
raw_api_responses & klines
    ↓
GapReport & DownloadResult
    ↓
ingestion_runs audit trail
    ↓
PRAGMA integrity_check & foreign_key_check
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock

from decimal import Decimal

from trade_intelligence.binance.client import BinanceRestClient
from trade_intelligence.binance.exceptions import BinanceHttpError, BinanceTimeoutError
from trade_intelligence.binance.models import ExchangeInfo, Kline, ServerTime
from trade_intelligence.db.database import Database
from trade_intelligence.db.exceptions import DatabaseError
from trade_intelligence.klines.downloader import HistoricalKlineDownloader
from trade_intelligence.universe.sync import UniverseSyncer


def _build_mock_exchange_info(symbols_raw: list, server_time_ms: int = 1700000000000) -> ExchangeInfo:
    """Construct an ExchangeInfo model from raw symbol dictionaries."""
    data = {
        "timezone": "UTC",
        "serverTime": server_time_ms,
        "rateLimits": [],
        "exchangeFilters": [],
        "symbols": symbols_raw,
    }
    return ExchangeInfo.from_raw(data)


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


class TestSystemIntegrationOffline(unittest.TestCase):
    """End-to-end integration tests starting from a fresh SQLite database."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "system_integration.db")
        self.db = Database(self.db_path)
        self.db.connect()
        self.db.initialize()

        self.mock_client = MagicMock(spec=BinanceRestClient)
        self.mock_session = MagicMock()
        self.mock_session.hooks = {"response": []}
        self.mock_client._session = self.mock_session

        self.syncer = UniverseSyncer(self.db, self.mock_client)
        self.downloader = HistoricalKlineDownloader(
            db=self.db,
            client=self.mock_client,
            request_delay_ms=0.0,
            max_retries=3,
        )

    def tearDown(self):
        self.db.close()

    def test_clean_database_full_pipeline_universe_to_klines(self):
        """Step 1 & 6: Clean DB -> Universe Sync -> Kline Download -> Gap Analysis."""
        # 1. Universe Sync from scratch
        server_time = 1700000000000
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw=[
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "baseAssetPrecision": 8,
                    "quoteAsset": "USDT",
                    "quotePrecision": 8,
                    "quoteAssetPrecision": 8,
                    "isSpotTradingAllowed": True,
                    "isMarginTradingAllowed": True,
                },
                {
                    "symbol": "ETHUSDT",
                    "status": "TRADING",
                    "baseAsset": "ETH",
                    "baseAssetPrecision": 8,
                    "quoteAsset": "USDT",
                    "quotePrecision": 8,
                    "quoteAssetPrecision": 8,
                    "isSpotTradingAllowed": True,
                    "isMarginTradingAllowed": False,
                },
            ],
            server_time_ms=server_time,
        )

        sync_res = self.syncer.sync_spot_universe(quote_asset="USDT")
        self.assertEqual(sync_res.new_symbols, 2)
        self.assertEqual(sync_res.records_stored, 2)

        # Confirm symbols exist in DB
        btc = self.db.get_symbol("BTCUSDT")
        self.assertIsNotNone(btc)
        self.assertEqual(btc["status"], "TRADING")

        # 2. Historical Kline Download for the universe-synced BTCUSDT
        download_server_time = 1700000000000 + 100 * 3600000
        self.mock_client.get_server_time.return_value = ServerTime.from_raw({"serverTime": download_server_time})

        base_time = 1700000000000 - (1700000000000 % 3600000)
        mock_klines = [_make_mock_kline(base_time + i * 3600000) for i in range(25)]
        self.mock_client.get_klines.return_value = mock_klines

        dl_res = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 24 * 3600000,
            resume=False,
        )

        self.assertEqual(dl_res.status, "completed")
        self.assertEqual(dl_res.records_stored, 25)
        self.assertEqual(dl_res.gap_report.total_actual_candles, 25)
        self.assertEqual(dl_res.gap_report.total_missing_candles, 0)
        self.assertEqual(dl_res.gap_report.coverage_ratio, 1.0)

        # 3. Database Integrity & Foreign Key Check
        conn = self.db.connection
        fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
        self.assertEqual(len(fk_violations), 0)
        integrity = conn.execute("PRAGMA integrity_check;").fetchone()
        self.assertEqual(integrity[0], "ok")

    def test_lifecycle_integration_trading_delisted_relisted_download(self):
        """Step 2: TRADING -> DELISTED -> TRADING -> Kline Download."""
        t1 = 1700000000000
        # 1. Initial sync: BTCUSDT TRADING
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw=[{
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "baseAssetPrecision": 8,
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": True,
            }],
            server_time_ms=t1,
        )
        self.syncer.sync_spot_universe()
        sym_initial = self.db.get_symbol("BTCUSDT")
        initial_created_at = sym_initial["created_at"]

        # 2. Delisting sync: BTCUSDT absent from exchangeInfo
        t2 = t1 + 86400000
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw=[],
            server_time_ms=t2,
        )
        sync2 = self.syncer.sync_spot_universe()
        self.assertEqual(sync2.delisted_symbols, 1)

        sym_delisted = self.db.get_symbol("BTCUSDT")
        self.assertEqual(sym_delisted["status"], "DELISTED")
        self.assertEqual(sym_delisted["created_at"], initial_created_at)

        # 2b. Verify downloader can ingest historical klines even while symbol is DELISTED
        base_time = 1700000000000 - (1700000000000 % 3600000)
        self.mock_client.get_server_time.return_value = ServerTime.from_raw({"serverTime": t2 + 1000000})
        self.mock_client.get_klines.return_value = [_make_mock_kline(base_time)]
        dl_delisted = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 3600000,
            resume=False,
        )
        self.assertEqual(dl_delisted.records_stored, 1)
        self.assertEqual(len(self.db.connection.execute("PRAGMA foreign_key_check;").fetchall()), 0)

        # 3. Re-listing sync: BTCUSDT reappears as TRADING
        t3 = t2 + 86400000
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw=[{
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "baseAssetPrecision": 8,
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": True,
            }],
            server_time_ms=t3,
        )
        sync3 = self.syncer.sync_spot_universe()
        self.assertEqual(sync3.status_transitions, 1)

        sym_relisted = self.db.get_symbol("BTCUSDT")
        self.assertEqual(sym_relisted["status"], "TRADING")
        self.assertEqual(sym_relisted["created_at"], initial_created_at)

        # Verify symbol status event ledger
        events = self.db.query_symbol_status_events("BTCUSDT", limit=10)
        # Event 1 (new baseline): TRADING
        # Event 2: DELISTED
        # Event 3: TRADING
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["status"], "TRADING")
        self.assertEqual(events[1]["status"], "DELISTED")
        self.assertEqual(events[2]["status"], "TRADING")

        # 4. Downloader successfully operates on the relisted symbol
        self.mock_client.get_server_time.return_value = ServerTime.from_raw({"serverTime": t3 + 1000000})
        self.mock_client.get_klines.return_value = [_make_mock_kline(base_time + 3600000)]

        dl_res = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time + 3600000,
            end_time=base_time + 2 * 3600000,
            resume=False,
        )
        self.assertEqual(dl_res.records_stored, 1)

        # Foreign key integrity check
        conn = self.db.connection
        self.assertEqual(len(conn.execute("PRAGMA foreign_key_check;").fetchall()), 0)
        self.assertEqual(conn.execute("PRAGMA integrity_check;").fetchone()[0], "ok")

    def test_deliberate_mid_download_failure_rollback_and_resume(self):
        """Step 3: Multi-page download fails on page 2, rolls back, then resumes cleanly."""
        base_time = 1700000000000 - (1700000000000 % 3600000)
        server_time = base_time + 3000 * 3600000
        self.mock_client.get_server_time.return_value = ServerTime.from_raw({"serverTime": server_time})

        # Pre-sync symbol in universe
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw=[{
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "baseAssetPrecision": 8,
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": True,
            }],
            server_time_ms=base_time,
        )
        self.syncer.sync_spot_universe()

        # Page 1: 1000 candles. Page 2: raises database error during batch_upsert_klines
        page1 = [_make_mock_kline(base_time + i * 3600000) for i in range(1000)]
        page2_start = page1[-1].close_time_ms + 1
        page2 = [_make_mock_kline(page2_start + i * 3600000) for i in range(500)]

        self.mock_client.get_klines.side_effect = [page1, page2]

        orig_upsert = self.db.batch_upsert_klines
        call_count = 0

        def failing_upsert(kline_rows):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise DatabaseError("Simulated disk error on page 2 batch upsert")
            return orig_upsert(kline_rows)

        self.db.batch_upsert_klines = failing_upsert

        with self.assertRaises(DatabaseError) as ctx:
            self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=base_time,
                end_time=base_time + 1500 * 3600000,
                resume=False,
            )

        self.assertIn("Simulated disk error", str(ctx.exception))

        # Restore original upsert
        self.db.batch_upsert_klines = orig_upsert

        # Verify Page 1 survived in database
        stored_p1 = self.db.query_klines_range("BTCUSDT", "1h", base_time, base_time + 2000 * 3600000)
        self.assertEqual(len(stored_p1), 1000)

        # Check raw response provenance:
        # Page 1 raw response is stored, and Page 2 raw response was committed before upsert failed
        conn = self.db.connection
        raw_rows = conn.execute("SELECT id, endpoint FROM raw_api_responses;").fetchall()
        # 1 from universe sync + 2 from download pages = 3 total
        self.assertEqual(len(raw_rows), 3)

        # Integrity and FK check must be clean (no orphan klines)
        self.assertEqual(len(conn.execute("PRAGMA foreign_key_check;").fetchall()), 0)
        self.assertEqual(conn.execute("PRAGMA integrity_check;").fetchone()[0], "ok")

        # Now resume from persisted state
        self.mock_client.get_klines.side_effect = [page2]
        resume_res = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 1500 * 3600000,
            resume=True,
        )

        self.assertEqual(resume_res.status, "completed")
        self.assertEqual(resume_res.records_stored, 500)
        self.assertEqual(resume_res.effective_start_time, page2_start)

        # All 1500 candles are present without duplicate or missing rows
        all_stored = self.db.query_klines_range("BTCUSDT", "1h", base_time, base_time + 2000 * 3600000)
        self.assertEqual(len(all_stored), 1500)

    def test_provenance_full_chain_audit(self):
        """Step 4: Strict audit of kline -> raw_response_id -> raw_api_responses -> ingestion_runs."""
        base_time = 1700000000000 - (1700000000000 % 3600000)
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw=[{
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "baseAssetPrecision": 8,
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": True,
            }],
            server_time_ms=base_time,
        )
        self.syncer.sync_spot_universe()

        # Multi-page download (Page 1: 1000, Page 2: 200)
        server_time = base_time + 2000 * 3600000
        self.mock_client.get_server_time.return_value = ServerTime.from_raw({"serverTime": server_time})

        page1 = [_make_mock_kline(base_time + i * 3600000) for i in range(1000)]
        page2_start = page1[-1].close_time_ms + 1
        page2 = [_make_mock_kline(page2_start + i * 3600000) for i in range(200)]
        self.mock_client.get_klines.side_effect = [page1, page2]

        dl_res = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 1200 * 3600000,
            resume=False,
        )

        conn = self.db.connection
        # Query all klines
        klines = conn.execute("SELECT symbol, open_time, raw_response_id FROM klines;").fetchall()
        self.assertEqual(len(klines), 1200)

        # Collect raw response IDs from klines
        raw_ids_in_klines = {k["raw_response_id"] for k in klines}
        self.assertEqual(len(raw_ids_in_klines), 2, "1200 candles over 2 pages must reference exactly 2 raw responses")

        # Query raw responses
        for raw_id in raw_ids_in_klines:
            raw_row = conn.execute("SELECT * FROM raw_api_responses WHERE id = ?;", (raw_id,)).fetchone()
            self.assertIsNotNone(raw_row)
            self.assertEqual(raw_row["endpoint"], "/api/v3/klines")
            self.assertEqual(raw_row["ingestion_run_id"], dl_res.run_id)

            # Validate SHA-256 hash
            import hashlib
            computed_hash = hashlib.sha256(raw_row["response_json"].encode("utf-8")).hexdigest()
            self.assertEqual(raw_row["content_hash"], computed_hash)

        # Ingestion run record
        run_row = conn.execute("SELECT * FROM ingestion_runs WHERE id = ?;", (dl_res.run_id,)).fetchone()
        self.assertIsNotNone(run_row)
        self.assertEqual(run_row["run_type"], "kline_download")
        self.assertEqual(run_row["status"], "completed")
        self.assertEqual(run_row["records_stored"], 1200)

        # Check zero orphaned FK references
        fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
        self.assertEqual(len(fk_violations), 0)

    def test_idempotent_repeated_download_preserves_state(self):
        """Step 5: Running the exact same historical download twice preserves state and counts."""
        base_time = 1700000000000 - (1700000000000 % 3600000)
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw=[{
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "baseAssetPrecision": 8,
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": True,
            }],
            server_time_ms=base_time,
        )
        self.syncer.sync_spot_universe()

        server_time = base_time + 100 * 3600000
        self.mock_client.get_server_time.return_value = ServerTime.from_raw({"serverTime": server_time})

        batch = [_make_mock_kline(base_time + i * 3600000) for i in range(50)]
        self.mock_client.get_klines.return_value = batch

        # Run 1
        res1 = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 50 * 3600000,
            resume=False,
        )
        self.assertEqual(res1.records_stored, 50)

        conn = self.db.connection
        cnt1 = conn.execute("SELECT COUNT(*) AS c FROM klines;").fetchone()["c"]
        self.assertEqual(cnt1, 50)

        # Run 2: exact same range, forced re-download (resume=False)
        res2 = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time,
            end_time=base_time + 50 * 3600000,
            resume=False,
        )
        self.assertEqual(res2.records_stored, 50)

        cnt2 = conn.execute("SELECT COUNT(*) AS c FROM klines;").fetchone()["c"]
        self.assertEqual(cnt2, 50, "Row count must remain exactly 50 (no duplicate candles)")

        # Verify integrity and foreign keys
        self.assertEqual(len(conn.execute("PRAGMA foreign_key_check;").fetchall()), 0)
        self.assertEqual(conn.execute("PRAGMA integrity_check;").fetchone()[0], "ok")

    def test_cross_phase_contracts_and_error_handling(self):
        """Verify cross-phase contracts: batch/single parity, non-retryable 400, transient retry."""
        base_time = 1700000000000 - (1700000000000 % 3600000)
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw=[{
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "baseAssetPrecision": 8,
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": True,
            }],
            server_time_ms=base_time,
        )
        self.syncer.sync_spot_universe()

        # 1. Parity between single-row insert_kline and batch_upsert_klines
        self.db.insert_kline(
            symbol="BTCUSDT",
            interval="1h",
            open_time=base_time,
            open_price=Decimal("40000.00"),
            high_price=Decimal("41000.00"),
            low_price=Decimal("39000.00"),
            close_price=Decimal("40500.00"),
            volume=Decimal("100.0"),
            close_time=base_time + 3599999,
            quote_asset_volume=Decimal("4050000.0"),
            number_of_trades=500,
            taker_buy_base_volume=Decimal("50.0"),
            taker_buy_quote_volume=Decimal("2025000.0"),
        )
        row_single = self.db.query_klines_range("BTCUSDT", "1h", base_time, base_time + 3600000)[0]
        self.assertEqual(Decimal(str(row_single["close_price"])), Decimal("40500.00"))

        # Now batch upsert updates that exact same candle
        self.db.batch_upsert_klines([{
            "symbol": "BTCUSDT",
            "interval": "1h",
            "open_time": base_time,
            "open_price": Decimal("40000.00"),
            "high_price": Decimal("41500.00"),
            "low_price": Decimal("39000.00"),
            "close_price": Decimal("41200.00"),
            "volume": Decimal("150.0"),
            "close_time": base_time + 3599999,
            "quote_asset_volume": Decimal("6180000.0"),
            "number_of_trades": 750,
            "taker_buy_base_volume": Decimal("75.0"),
            "taker_buy_quote_volume": Decimal("3090000.0"),
        }])
        row_upserted = self.db.query_klines_range("BTCUSDT", "1h", base_time, base_time + 3600000)[0]
        self.assertEqual(Decimal(str(row_upserted["close_price"])), Decimal("41200.00"))
        self.assertEqual(Decimal(str(row_upserted["volume"])), Decimal("150.0"))
        self.assertEqual(row_upserted["number_of_trades"], 750)

        # 2. Non-retryable Binance error (e.g. 400 Bad Request) must NOT be retried
        self.mock_client.get_server_time.return_value = ServerTime.from_raw({"serverTime": base_time + 10 * 3600000})
        self.mock_client.get_klines.side_effect = BinanceHttpError(
            status_code=400,
            message="Illegal characters found in parameter 'symbol'",
        )

        with self.assertRaises(BinanceHttpError) as ctx:
            self.downloader.download_historical_klines(
                symbol="BTCUSDT",
                interval="1h",
                start_time=base_time + 3600000,
                end_time=base_time + 2 * 3600000,
                resume=False,
            )
        self.assertEqual(ctx.exception.status_code, 400)
        # Verify client get_klines was called exactly ONCE (never retried)
        self.assertEqual(self.mock_client.get_klines.call_count, 1)

        # Verify ingestion_runs was marked 'failed'
        conn = self.db.connection
        last_run = conn.execute("SELECT * FROM ingestion_runs ORDER BY id DESC LIMIT 1;").fetchone()
        self.assertEqual(last_run["status"], "failed")
        self.assertIn("Illegal characters", last_run["error_message"])

        # 3. Transient error (e.g. Timeout) IS retried and succeeds
        self.mock_client.get_klines.reset_mock()
        mock_kline = _make_mock_kline(base_time + 3600000)
        self.mock_client.get_klines.side_effect = [
            BinanceTimeoutError("Connection timed out"),
            [mock_kline],
        ]

        res = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=base_time + 3600000,
            end_time=base_time + 2 * 3600000,
            resume=False,
        )
        self.assertEqual(res.status, "completed")
        self.assertEqual(res.records_stored, 1)
        self.assertEqual(self.mock_client.get_klines.call_count, 2)



@unittest.skipUnless(
    os.getenv("RUN_LIVE_TESTS") == "1",
    "Live integration tests disabled. Set RUN_LIVE_TESTS=1 to run.",
)
class TestSystemIntegrationLive(unittest.TestCase):
    """End-to-end live integration starting from a clean SQLite database."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "system_integration_live.db")
        self.db = Database(self.db_path)
        self.db.connect()
        self.db.initialize()

        self.client = BinanceRestClient()
        self.syncer = UniverseSyncer(self.db, self.client)
        self.downloader = HistoricalKlineDownloader(
            db=self.db,
            client=self.client,
            request_delay_ms=250.0,
            max_retries=3,
        )

    def tearDown(self):
        self.client.close()
        self.db.close()

    def test_live_clean_database_universe_to_historical_download(self):
        """Execute live: Clean DB -> Universe Sync -> Multi-Page Download -> Tail Resume."""
        # 1. Authoritative Spot Universe Sync against Binance API
        sync_result = self.syncer.sync_spot_universe(quote_asset="USDT")
        self.assertGreater(sync_result.records_stored, 100, "Should store > 100 USDT spot symbols")

        btc = self.db.get_symbol("BTCUSDT")
        self.assertIsNotNone(btc, "BTCUSDT must be discovered and stored")
        self.assertEqual(btc["status"], "TRADING")

        # 2. Historical Download for BTCUSDT: 2 pages (1200 candles) safely in past
        server_time = self.client.get_server_time().server_time_ms
        h_ms = 3600 * 1000
        raw_end = server_time - (2 * 24 * h_ms)
        end_time = raw_end - (raw_end % h_ms)
        start_time = end_time - (1200 * h_ms)

        dl_result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=start_time,
            end_time=end_time,
            resume=False,
        )

        self.assertEqual(dl_result.status, "completed")
        self.assertGreaterEqual(dl_result.records_stored, 1000)
        self.assertGreaterEqual(dl_result.pages_fetched, 2)

        # 3. Provenance & DB Checks
        conn = self.db.connection
        fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
        self.assertEqual(len(fk_violations), 0)
        integrity = conn.execute("PRAGMA integrity_check;").fetchone()
        self.assertEqual(integrity[0], "ok")

        # 4. Tail Resume: Run over same range
        resume_result = self.downloader.download_historical_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=start_time,
            end_time=end_time,
            resume=True,
        )
        self.assertEqual(resume_result.status, "already_up_to_date")
        self.assertEqual(resume_result.records_stored, 0)
        self.assertEqual(resume_result.pages_fetched, 0)
