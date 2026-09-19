"""Live integration tests for Binance Spot universe synchronization.

Hits the real Binance public exchangeInfo endpoint and persists real USDT
symbols into a temporary database.
"""

import os
import tempfile
import unittest

from trade_intelligence.binance.client import BinanceRestClient
from trade_intelligence.db.database import Database
from trade_intelligence.universe.sync import SyncResult, UniverseSyncer


@unittest.skipUnless(
    os.getenv("RUN_LIVE_TESTS") == "1",
    "Live Binance integration tests are disabled by default. Set RUN_LIVE_TESTS=1 to run.",
)
class TestUniverseSyncLiveIntegration(unittest.TestCase):
    """Live universe synchronization tests against Binance Spot REST API."""

    @classmethod
    def setUpClass(cls):
        cls.client = BinanceRestClient(timeout=30.0)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_live_universe.db")
        self.db = Database(self.db_path)
        self.db.connect()
        self.db.initialize()
        self.syncer = UniverseSyncer(self.db, self.client)

    def tearDown(self):
        self.db.close()

    def test_live_universe_sync_full_lifecycle(self):
        """Perform live sync from Binance Spot exchangeInfo into temporary database."""
        # 1. First sync
        res1 = self.syncer.sync_spot_universe(quote_asset="USDT")
        self.assertIsInstance(res1, SyncResult)
        self.assertGreater(res1.records_fetched, 1000)
        self.assertGreater(res1.records_stored, 300)
        self.assertEqual(res1.new_symbols, res1.records_stored)

        # 2. Check BTCUSDT and ETHUSDT exist in symbols table
        btc = self.db.get_symbol("BTCUSDT")
        self.assertIsNotNone(btc)
        self.assertEqual(btc["quote_asset"], "USDT")
        self.assertEqual(btc["status"], "TRADING")
        self.assertTrue(btc["is_spot_trading_allowed"])

        eth = self.db.get_symbol("ETHUSDT")
        self.assertIsNotNone(eth)
        self.assertEqual(eth["quote_asset"], "USDT")

        # 3. Check foreign keys and SQLite integrity
        conn = self.db.connection
        fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        self.assertEqual(len(fk_violations), 0)
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        self.assertEqual(integrity, "ok")

        # 4. Immediate second sync should be completely idempotent (0 new status events)
        res2 = self.syncer.sync_spot_universe(quote_asset="USDT")
        self.assertEqual(res2.new_symbols, 0)
        self.assertEqual(res2.delisted_symbols, 0)
        self.assertEqual(res2.status_transitions, 0)
