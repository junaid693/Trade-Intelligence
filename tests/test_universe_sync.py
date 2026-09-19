"""Comprehensive tests for Binance Spot market universe synchronization (Phase 1.4.1).

Tests verify:
- Initial universe discovery & baseline event generation.
- Strict idempotency (zero duplicate status events on repeated identical sync).
- Status and permission transitions.
- Non-USDT pair filtering (only target quote asset persisted to symbols table).
- Delisted/missing symbol detection without row deletion (survivorship-bias protection).
- Dual transaction boundary isolation: raw response preserved when symbol upsert fails.
- SQLite foreign key integrity and PRAGMA integrity_check.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from trade_intelligence.binance.client import BinanceRestClient
from trade_intelligence.binance.models import ExchangeInfo
from trade_intelligence.db.database import Database
from trade_intelligence.db.exceptions import DatabaseError
from trade_intelligence.universe.sync import SyncResult, UniverseSyncer


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


class TestUniverseSync(unittest.TestCase):
    """Test suite for UniverseSyncer."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_universe.db")
        self.db = Database(self.db_path)
        self.db.connect()
        self.db.initialize()
        self.mock_client = MagicMock(spec=BinanceRestClient)
        self.syncer = UniverseSyncer(self.db, self.mock_client)

    def tearDown(self):
        self.db.close()

    def test_universe_sync_first_run_success(self):
        """First sync discovers new symbols and creates baseline status events."""
        symbols_raw = [
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
            {
                "symbol": "BNBBTC",  # Non-USDT pair, should be filtered out
                "status": "TRADING",
                "baseAsset": "BNB",
                "baseAssetPrecision": 8,
                "quoteAsset": "BTC",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": True,
            },
        ]
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw, server_time_ms=1700000000000
        )

        res = self.syncer.sync_spot_universe(quote_asset="USDT")

        self.assertIsInstance(res, SyncResult)
        self.assertEqual(res.records_fetched, 3)
        self.assertEqual(res.records_stored, 2)
        self.assertEqual(res.new_symbols, 2)
        self.assertEqual(res.status_transitions, 0)
        self.assertEqual(res.delisted_symbols, 0)

        # Check symbols table
        symbols = self.db.get_symbols()
        self.assertEqual(len(symbols), 2)
        sym_names = [s["symbol"] for s in symbols]
        self.assertIn("BTCUSDT", sym_names)
        self.assertIn("ETHUSDT", sym_names)
        self.assertNotIn("BNBBTC", sym_names)

        # Check status events table
        btc_events = self.db.query_symbol_status_events("BTCUSDT")
        self.assertEqual(len(btc_events), 1)
        self.assertEqual(btc_events[0]["status"], "TRADING")
        self.assertTrue(btc_events[0]["is_spot_trading_allowed"])
        self.assertEqual(btc_events[0]["raw_response_id"], res.raw_response_id)

        # Check raw response
        conn = self.db.connection
        raw_row = conn.execute("SELECT * FROM raw_api_responses WHERE id = ?", (res.raw_response_id,)).fetchone()
        self.assertIsNotNone(raw_row)
        self.assertEqual(raw_row["endpoint"], "/api/v3/exchangeInfo")
        self.assertEqual(raw_row["ingestion_run_id"], res.run_id)

        # Check ingestion run
        run_row = conn.execute("SELECT * FROM ingestion_runs WHERE id = ?", (res.run_id,)).fetchone()
        self.assertIsNotNone(run_row)
        self.assertEqual(run_row["run_type"], "exchange_info_sync")
        self.assertEqual(run_row["status"], "completed")
        self.assertEqual(run_row["records_fetched"], 3)
        self.assertEqual(run_row["records_stored"], 2)

    def test_universe_sync_idempotency(self):
        """Consecutive syncs with unchanged exchangeInfo insert 0 new status events."""
        symbols_raw = [
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
        ]
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw, server_time_ms=1700000000000
        )

        res1 = self.syncer.sync_spot_universe()
        self.assertEqual(res1.new_symbols, 1)

        btc1 = self.db.get_symbol("BTCUSDT")
        created_at_initial = btc1["created_at"]
        self.assertEqual(btc1["updated_at"], 1700000000000)
        self.assertEqual(len(self.db.query_symbol_status_events("BTCUSDT")), 1)

        # Second sync with same data but newer server time
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw, server_time_ms=1700000005000
        )

        res2 = self.syncer.sync_spot_universe()
        self.assertEqual(res2.new_symbols, 0)
        self.assertEqual(res2.status_transitions, 0)
        self.assertEqual(res2.delisted_symbols, 0)

        # Verify created_at unchanged, updated_at updated
        btc2 = self.db.get_symbol("BTCUSDT")
        self.assertEqual(btc2["created_at"], created_at_initial)
        self.assertEqual(btc2["updated_at"], 1700000005000)

        # Verify 0 duplicate status events
        btc_events_after = self.db.query_symbol_status_events("BTCUSDT")
        self.assertEqual(len(btc_events_after), 1)

    def test_universe_sync_status_transition(self):
        """Status change emits exactly one transition event."""
        symbols_raw_v1 = [
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
        ]
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw_v1, server_time_ms=1700000000000
        )
        self.syncer.sync_spot_universe()

        # Update status to BREAK
        symbols_raw_v2 = [
            {
                "symbol": "BTCUSDT",
                "status": "BREAK",
                "baseAsset": "BTC",
                "baseAssetPrecision": 8,
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": True,
            },
        ]
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw_v2, server_time_ms=1700000010000
        )
        res = self.syncer.sync_spot_universe()

        self.assertEqual(res.status_transitions, 1)
        self.assertEqual(res.new_symbols, 0)

        btc = self.db.get_symbol("BTCUSDT")
        self.assertEqual(btc["status"], "BREAK")

        events = self.db.query_symbol_status_events("BTCUSDT")
        self.assertEqual(len(events), 2)
        # Newest event first
        self.assertEqual(events[0]["status"], "BREAK")
        self.assertEqual(events[0]["effective_at"], 1700000010000)
        self.assertEqual(events[1]["status"], "TRADING")

    def test_universe_sync_permission_transition(self):
        """Spot permission change emits transition event."""
        symbols_raw_v1 = [
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
        ]
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(symbols_raw_v1)
        self.syncer.sync_spot_universe()

        # Revoke spot permission
        symbols_raw_v2 = [
            {
                "symbol": "ETHUSDT",
                "status": "TRADING",
                "baseAsset": "ETH",
                "baseAssetPrecision": 8,
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "isSpotTradingAllowed": False,
                "isMarginTradingAllowed": False,
            },
        ]
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(symbols_raw_v2)
        res = self.syncer.sync_spot_universe()

        self.assertEqual(res.status_transitions, 1)
        eth = self.db.get_symbol("ETHUSDT")
        self.assertFalse(eth["is_spot_trading_allowed"])

        events = self.db.query_symbol_status_events("ETHUSDT")
        self.assertEqual(len(events), 2)
        self.assertFalse(events[0]["is_spot_trading_allowed"])
        self.assertTrue(events[1]["is_spot_trading_allowed"])

    def test_universe_sync_delisted_symbol_preservation(self):
        """Missing symbols are preserved in database as DELISTED (no deletion)."""
        symbols_raw_v1 = [
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
                "symbol": "OLDUSDT",
                "status": "TRADING",
                "baseAsset": "OLD",
                "baseAssetPrecision": 8,
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": False,
            },
        ]
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(symbols_raw_v1)
        self.syncer.sync_spot_universe()

        # V2: OLDUSDT disappears from exchangeInfo
        symbols_raw_v2 = [
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
        ]
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(symbols_raw_v2)
        res = self.syncer.sync_spot_universe()

        self.assertEqual(res.delisted_symbols, 1)

        # OLDUSDT must still exist in symbols table!
        old_sym = self.db.get_symbol("OLDUSDT")
        self.assertIsNotNone(old_sym)
        self.assertEqual(old_sym["status"], "DELISTED")
        self.assertFalse(old_sym["is_spot_trading_allowed"])

        # Status event emitted for DELISTED
        events = self.db.query_symbol_status_events("OLDUSDT")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["status"], "DELISTED")
        self.assertFalse(events[0]["is_spot_trading_allowed"])

    def test_universe_sync_relisted_symbol(self):
        """Symbol that was DELISTED can reappear and transition back to TRADING seamlessly."""
        # 1. First discovery: RELISTUSDT is TRADING
        symbols_raw_v1 = [
            {
                "symbol": "RELISTUSDT",
                "status": "TRADING",
                "baseAsset": "RELIST",
                "baseAssetPrecision": 8,
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": False,
            },
        ]
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw_v1, server_time_ms=1700000000000
        )
        res1 = self.syncer.sync_spot_universe()
        self.assertEqual(res1.new_symbols, 1)

        initial_row = self.db.get_symbol("RELISTUSDT")
        created_at_first_seen = initial_row["created_at"]
        self.assertEqual(initial_row["status"], "TRADING")
        self.assertTrue(initial_row["is_spot_trading_allowed"])

        # 2. Sync V2: RELISTUSDT disappears from exchangeInfo -> marked DELISTED
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            [], server_time_ms=1700000010000
        )
        res2 = self.syncer.sync_spot_universe()
        self.assertEqual(res2.delisted_symbols, 1)

        delisted_row = self.db.get_symbol("RELISTUSDT")
        self.assertIsNotNone(delisted_row)
        self.assertEqual(delisted_row["status"], "DELISTED")
        self.assertFalse(delisted_row["is_spot_trading_allowed"])
        self.assertEqual(delisted_row["created_at"], created_at_first_seen)

        events_v2 = self.db.query_symbol_status_events("RELISTUSDT")
        self.assertEqual(len(events_v2), 2)
        self.assertEqual(events_v2[0]["status"], "DELISTED")
        self.assertFalse(events_v2[0]["is_spot_trading_allowed"])
        self.assertEqual(events_v2[1]["status"], "TRADING")

        # 3. Sync V3: RELISTUSDT reappears in exchangeInfo with TRADING and spot enabled
        symbols_raw_v3 = [
            {
                "symbol": "RELISTUSDT",
                "status": "TRADING",
                "baseAsset": "RELIST",
                "baseAssetPrecision": 8,
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "quoteAssetPrecision": 8,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": True,
            },
        ]
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw_v3, server_time_ms=1700000020000
        )
        res3 = self.syncer.sync_spot_universe()
        self.assertEqual(res3.new_symbols, 0)
        self.assertEqual(res3.status_transitions, 1)
        self.assertEqual(res3.delisted_symbols, 0)

        relisted_row = self.db.get_symbol("RELISTUSDT")
        self.assertEqual(relisted_row["status"], "TRADING")
        self.assertTrue(relisted_row["is_spot_trading_allowed"])
        self.assertTrue(relisted_row["is_margin_trading_allowed"])
        self.assertEqual(relisted_row["created_at"], created_at_first_seen)
        self.assertEqual(relisted_row["updated_at"], 1700000020000)

        # Verify exactly one new transition event was recorded (total 3 events now)
        events_v3 = self.db.query_symbol_status_events("RELISTUSDT")
        self.assertEqual(len(events_v3), 3)
        self.assertEqual(events_v3[0]["status"], "TRADING")
        self.assertTrue(events_v3[0]["is_spot_trading_allowed"])
        self.assertEqual(events_v3[0]["effective_at"], 1700000020000)

        # 4. Sync V4: Repeated sync with unchanged V3 data -> 0 new events, no duplicates
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(
            symbols_raw_v3, server_time_ms=1700000030000
        )
        res4 = self.syncer.sync_spot_universe()
        self.assertEqual(res4.new_symbols, 0)
        self.assertEqual(res4.status_transitions, 0)
        self.assertEqual(res4.delisted_symbols, 0)

        events_v4 = self.db.query_symbol_status_events("RELISTUSDT")
        self.assertEqual(len(events_v4), 3)

        # Verify foreign keys across the database
        conn = self.db.connection
        self.assertEqual(len(conn.execute("PRAGMA foreign_key_check").fetchall()), 0)
        self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_universe_sync_transaction_isolation_preserves_raw_response_on_upsert_failure(self):
        """Raw response is permanently saved even if symbol batch upsert fails."""
        symbols_raw = [
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
        ]
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(symbols_raw)

        # Mock database batch_upsert_universe_state to fail
        self.db.batch_upsert_universe_state = MagicMock(side_effect=DatabaseError("Simulated DB write failure"))

        with self.assertRaises(DatabaseError):
            self.syncer.sync_spot_universe()

        # Verify raw response WAS saved (Transaction Boundary 1)
        conn = self.db.connection
        raw_rows = conn.execute("SELECT * FROM raw_api_responses").fetchall()
        self.assertEqual(len(raw_rows), 1)
        self.assertEqual(raw_rows[0]["endpoint"], "/api/v3/exchangeInfo")

        # Verify ingestion run recorded failure
        run_row = conn.execute("SELECT * FROM ingestion_runs WHERE id = ?", (raw_rows[0]["ingestion_run_id"],)).fetchone()
        self.assertEqual(run_row["status"], "failed")
        self.assertIn("Simulated DB write failure", run_row["error_message"])

        # Verify symbols table has 0 rows (Transaction Boundary 2 never committed)
        self.assertEqual(len(self.db.get_symbols()), 0)

    def test_universe_sync_invalid_quote_asset_raises(self):
        """Empty or invalid quote_asset raises ValueError."""
        with self.assertRaises(ValueError):
            self.syncer.sync_spot_universe(quote_asset="")

        with self.assertRaises(ValueError):
            self.syncer.sync_spot_universe(quote_asset="   ")

    def test_database_integrity_after_sync(self):
        """Database integrity check and foreign key check pass with 0 errors."""
        symbols_raw = [
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
        ]
        self.mock_client.get_exchange_info.return_value = _build_mock_exchange_info(symbols_raw)
        self.syncer.sync_spot_universe()

        conn = self.db.connection
        fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        self.assertEqual(len(fk_violations), 0)

        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        self.assertEqual(integrity, "ok")


if __name__ == "__main__":
    unittest.main()
