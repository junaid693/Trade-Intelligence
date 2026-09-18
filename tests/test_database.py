"""Comprehensive tests for the Trade Intelligence SQLite database foundation.

Tests cover all Phase 1.3.2 requirements:
- Database initialization and schema creation
- Foreign-key enforcement
- Primary-key uniqueness and WITHOUT ROWID behavior
- Decimal value round-trip (TEXT storage)
- Timestamp round-trip (INTEGER storage)
- All repository methods (symbols, status events, raw responses,
  ingestion runs, klines, ticker snapshots)
- Kline upsert semantics
- Range and latest-N kline queries with correct ordering
- Rollback behavior on failed transactions
- Data persistence across database close/reopen
- EXPLAIN QUERY PLAN verification for kline queries
"""

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from trade_intelligence.db.database import Database, DEFAULT_PRAGMAS
from trade_intelligence.db.exceptions import (
    DatabaseError,
    DatabaseInitError,
    DatabaseIntegrityError,
)
from trade_intelligence.db.schema import SCHEMA_SQL, SCHEMA_VERSION


def _make_test_db(tmp_dir: str) -> Database:
    """Create and initialize a Database in a temp directory."""
    db_path = os.path.join(tmp_dir, "test.db")
    db = Database(db_path)
    db.connect()
    db.initialize()
    return db


def _insert_test_symbol(db: Database, symbol: str = "BTCUSDT") -> None:
    """Insert a standard test symbol."""
    db.insert_symbol(
        symbol=symbol,
        base_asset="BTC" if symbol == "BTCUSDT" else symbol[:3],
        quote_asset="USDT",
        status="TRADING",
        is_spot_trading_allowed=True,
        is_margin_trading_allowed=False,
        base_asset_precision=8,
        quote_asset_precision=8,
        updated_at=1700000000000,
    )


def _insert_test_kline(
    db: Database,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    open_time: int = 1700000000000,
    close_price: Decimal = Decimal("50500.12345678"),
) -> None:
    """Insert a standard test kline."""
    db.insert_kline(
        symbol=symbol,
        interval=interval,
        open_time=open_time,
        open_price=Decimal("50000.00000001"),
        high_price=Decimal("51000.99999999"),
        low_price=Decimal("49000.00000001"),
        close_price=close_price,
        volume=Decimal("123.45678900"),
        close_time=open_time + 3599999,
        quote_asset_volume=Decimal("6172839.50000000"),
        number_of_trades=15420,
        taker_buy_base_volume=Decimal("61.72839450"),
        taker_buy_quote_volume=Decimal("3086419.75000000"),
    )


class TestDatabaseInitialization(unittest.TestCase):
    """Tests for database initialization and connection management."""

    def test_database_initialization(self):
        """Verify database file is created and tables exist after initialization."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            db = Database(db_path)
            db.connect()
            db.initialize()

            # File should exist on disk
            self.assertTrue(os.path.exists(db_path))

            # Should be able to query tables
            conn = db.connection
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            ).fetchall()
            table_names = [row["name"] for row in rows]

            self.assertIn("klines", table_names)
            self.assertIn("symbols", table_names)
            self.assertIn("ingestion_runs", table_names)
            self.assertIn("raw_api_responses", table_names)
            self.assertIn("symbol_status_events", table_names)
            self.assertIn("ticker_snapshots", table_names)
            self.assertIn("schema_version", table_names)

            db.close()

    def test_schema_creation_all_tables(self):
        """Verify all 7 tables (6 data + schema_version) are created."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            try:
                conn = db.connection
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
                ).fetchall()
                # Filter out SQLite internal tables (e.g. sqlite_sequence for AUTOINCREMENT)
                table_names = sorted(
                    row["name"] for row in rows
                    if not row["name"].startswith("sqlite_")
                )

                expected = sorted([
                    "ingestion_runs",
                    "klines",
                    "raw_api_responses",
                    "schema_version",
                    "symbol_status_events",
                    "symbols",
                    "ticker_snapshots",
                ])
                self.assertEqual(table_names, expected)
            finally:
                db.close()

    def test_schema_version_tracking(self):
        """Verify schema version is recorded and retrievable."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            version = db.get_schema_version()
            self.assertEqual(version, SCHEMA_VERSION)
            self.assertEqual(version, 1)
            db.close()

    def test_schema_version_idempotent(self):
        """Verify calling initialize() twice does not duplicate version rows."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            db.initialize()  # Second call
            conn = db.connection
            count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM schema_version;"
            ).fetchone()["cnt"]
            self.assertEqual(count, 1)
            db.close()

    def test_context_manager(self):
        """Verify Database works as a context manager."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            with Database(db_path) as db:
                db.initialize()
                self.assertIsNotNone(db.connection)
            # After exit, connection should be None
            self.assertIsNone(db._conn)

    def test_connect_not_connected_error(self):
        """Verify accessing connection before connect() raises DatabaseError."""
        db = Database(":memory:")
        with self.assertRaises(DatabaseError):
            _ = db.connection

    def test_default_pragmas_applied(self):
        """Verify default pragmas are applied on connection."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            conn = db.connection

            fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
            self.assertEqual(fk, 1)

            journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            self.assertEqual(journal.upper(), "WAL")

            db.close()

    def test_custom_pragmas(self):
        """Verify custom pragmas override defaults (except foreign_keys)."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            db = Database(db_path, pragmas={"cache_size": -32000, "foreign_keys": 0})
            db.connect()

            conn = db.connection
            cache = conn.execute("PRAGMA cache_size;").fetchone()[0]
            self.assertEqual(cache, -32000)

            # foreign_keys must always be 1 regardless of caller
            fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
            self.assertEqual(fk, 1)

            db.close()


class TestForeignKeyEnforcement(unittest.TestCase):
    """Tests for foreign-key constraint enforcement."""

    def test_foreign_key_enforcement(self):
        """Verify FK violations raise DatabaseIntegrityError."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)

            # Inserting a status event for a non-existent symbol should fail
            with self.assertRaises(DatabaseIntegrityError):
                db.insert_symbol_status_event(
                    symbol="NONEXISTENT",
                    status="TRADING",
                    is_spot_trading_allowed=True,
                    effective_at=1700000000000,
                )

            db.close()

    def test_kline_fk_to_symbols(self):
        """Verify kline insert fails if symbol does not exist in symbols table."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)

            with self.assertRaises(DatabaseIntegrityError):
                _insert_test_kline(db, symbol="NOSYMBOL")

            db.close()

    def test_ticker_snapshot_fk_to_symbols(self):
        """Verify ticker snapshot insert fails if symbol does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)

            with self.assertRaises(DatabaseIntegrityError):
                db.insert_ticker_snapshot(
                    symbol="NOSYMBOL",
                    price=Decimal("50000"),
                    snapshot_at=1700000000000,
                )

            db.close()


class TestPrimaryKeyAndWithoutRowid(unittest.TestCase):
    """Tests for primary-key uniqueness and WITHOUT ROWID behavior."""

    def test_primary_key_uniqueness(self):
        """Verify duplicate PK insert raises DatabaseIntegrityError."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)
            _insert_test_kline(db)

            # Same PK should fail on plain insert
            with self.assertRaises(DatabaseIntegrityError):
                _insert_test_kline(db)

            db.close()

    def test_klines_without_rowid(self):
        """Verify the klines table is created WITH the WITHOUT ROWID clause."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            conn = db.connection

            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='klines';"
            ).fetchone()
            create_sql = row["sql"]
            self.assertIn("WITHOUT ROWID", create_sql.upper())

            db.close()


class TestDecimalAndTimestampRoundTrip(unittest.TestCase):
    """Tests for Decimal precision and timestamp integrity."""

    def test_decimal_round_trip(self):
        """Verify Decimal values survive write→read without precision loss."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            # Use extremely precise values that would fail with float
            precise_open = Decimal("0.00000001")
            precise_close = Decimal("99999999.99999999")
            precise_volume = Decimal("123456789.12345678")

            db.insert_kline(
                symbol="BTCUSDT",
                interval="1h",
                open_time=1700000000000,
                open_price=precise_open,
                high_price=Decimal("100000000.00000000"),
                low_price=Decimal("0.00000001"),
                close_price=precise_close,
                volume=precise_volume,
                close_time=1700003599999,
                quote_asset_volume=Decimal("5000000000.12345678"),
                number_of_trades=100,
                taker_buy_base_volume=Decimal("61728394.50000001"),
                taker_buy_quote_volume=Decimal("3086419750.00000001"),
            )

            results = db.query_klines_range("BTCUSDT", "1h", 1700000000000, 1700010000000)
            self.assertEqual(len(results), 1)

            kline = results[0]
            self.assertIsInstance(kline["open_price"], Decimal)
            self.assertIsInstance(kline["close_price"], Decimal)
            self.assertIsInstance(kline["volume"], Decimal)

            self.assertEqual(kline["open_price"], precise_open)
            self.assertEqual(kline["close_price"], precise_close)
            self.assertEqual(kline["volume"], precise_volume)

            # Verify the IEEE-754 problematic case: 0.1 + 0.2 == 0.3
            self.assertEqual(
                Decimal(str(kline["open_price"])),
                Decimal("0.00000001"),
            )

            db.close()

    def test_timestamp_round_trip(self):
        """Verify UTC epoch ms timestamps survive write→read as integers."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            open_time = 1700000000000
            close_time = 1700003599999

            _insert_test_kline(db, open_time=open_time)

            results = db.query_klines_range("BTCUSDT", "1h", open_time, open_time + 1)
            self.assertEqual(len(results), 1)

            kline = results[0]
            self.assertIsInstance(kline["open_time"], int)
            self.assertIsInstance(kline["close_time"], int)
            self.assertEqual(kline["open_time"], open_time)
            self.assertEqual(kline["close_time"], close_time)

            db.close()


class TestSymbolRepository(unittest.TestCase):
    """Tests for symbol insert/update operations."""

    def test_symbol_insert_update(self):
        """Verify symbol insert and subsequent update via INSERT OR REPLACE."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)

            # Initial insert
            db.insert_symbol(
                symbol="BTCUSDT",
                base_asset="BTC",
                quote_asset="USDT",
                status="TRADING",
                is_spot_trading_allowed=True,
                is_margin_trading_allowed=False,
                base_asset_precision=8,
                quote_asset_precision=8,
                updated_at=1700000000000,
            )

            conn = db.connection
            row = conn.execute(
                "SELECT * FROM symbols WHERE symbol = 'BTCUSDT';"
            ).fetchone()
            self.assertEqual(row["status"], "TRADING")
            self.assertEqual(row["is_spot_trading_allowed"], 1)

            # Update same symbol with changed status
            db.insert_symbol(
                symbol="BTCUSDT",
                base_asset="BTC",
                quote_asset="USDT",
                status="BREAK",
                is_spot_trading_allowed=False,
                is_margin_trading_allowed=False,
                base_asset_precision=8,
                quote_asset_precision=8,
                updated_at=1700001000000,
            )

            row = conn.execute(
                "SELECT * FROM symbols WHERE symbol = 'BTCUSDT';"
            ).fetchone()
            self.assertEqual(row["status"], "BREAK")
            self.assertEqual(row["is_spot_trading_allowed"], 0)
            self.assertEqual(row["updated_at"], 1700001000000)

            db.close()

    def test_symbol_update_preserves_child_rows(self):
        """Regression: updating a symbol must not destroy FK-linked child rows.

        INSERT OR REPLACE performs DELETE + INSERT internally, which would
        either violate FK constraints or (with CASCADE) destroy all klines,
        status events, and ticker snapshots for the symbol. The correct
        implementation uses INSERT ... ON CONFLICT DO UPDATE for an in-place
        update that preserves all foreign-key relationships.
        """
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)

            # 1. Insert symbol
            _insert_test_symbol(db, "BTCUSDT")

            # 2. Populate child rows referencing this symbol
            _insert_test_kline(db, symbol="BTCUSDT", open_time=1700000000000)
            _insert_test_kline(db, symbol="BTCUSDT", open_time=1700003600000)

            db.insert_symbol_status_event(
                symbol="BTCUSDT",
                status="TRADING",
                is_spot_trading_allowed=True,
                effective_at=1700000000000,
            )

            db.insert_ticker_snapshot(
                symbol="BTCUSDT",
                price=Decimal("67890.12345678"),
                snapshot_at=1700000000000,
            )

            # 3. Update the symbol (this is the operation under test)
            db.insert_symbol(
                symbol="BTCUSDT",
                base_asset="BTC",
                quote_asset="USDT",
                status="BREAK",
                is_spot_trading_allowed=False,
                is_margin_trading_allowed=False,
                base_asset_precision=8,
                quote_asset_precision=8,
                updated_at=1700005000000,
            )

            # 4. Verify the symbol was updated
            conn = db.connection
            sym_row = conn.execute(
                "SELECT * FROM symbols WHERE symbol = 'BTCUSDT';"
            ).fetchone()
            self.assertEqual(sym_row["status"], "BREAK")
            self.assertEqual(sym_row["is_spot_trading_allowed"], 0)
            self.assertEqual(sym_row["updated_at"], 1700005000000)

            # 5. Verify ALL child rows survived the update
            klines = db.query_klines_range(
                "BTCUSDT", "1h", 1700000000000, 1700010000000
            )
            self.assertEqual(len(klines), 2, "Klines must survive symbol update")

            events = conn.execute(
                "SELECT COUNT(*) AS cnt FROM symbol_status_events "
                "WHERE symbol = 'BTCUSDT';"
            ).fetchone()["cnt"]
            self.assertEqual(events, 1, "Status events must survive symbol update")

            snaps = db.query_ticker_snapshots("BTCUSDT")
            self.assertEqual(len(snaps), 1, "Ticker snapshots must survive symbol update")

            db.close()


class TestSymbolStatusEvents(unittest.TestCase):
    """Tests for symbol status event insertion."""

    def test_symbol_status_events_insertion(self):
        """Verify status events are appended and reference valid symbols."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            event_id = db.insert_symbol_status_event(
                symbol="BTCUSDT",
                status="TRADING",
                is_spot_trading_allowed=True,
                effective_at=1700000000000,
            )
            self.assertIsInstance(event_id, int)
            self.assertGreater(event_id, 0)

            # Insert a second event
            event_id_2 = db.insert_symbol_status_event(
                symbol="BTCUSDT",
                status="BREAK",
                is_spot_trading_allowed=False,
                effective_at=1700001000000,
            )
            self.assertGreater(event_id_2, event_id)

            conn = db.connection
            rows = conn.execute(
                "SELECT * FROM symbol_status_events WHERE symbol = 'BTCUSDT' ORDER BY effective_at;"
            ).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["status"], "TRADING")
            self.assertEqual(rows[1]["status"], "BREAK")

            db.close()


class TestRawApiResponses(unittest.TestCase):
    """Tests for raw API response storage."""

    def test_raw_api_response_storage(self):
        """Verify raw JSON is stored with computed SHA-256 hash."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)

            payload = json.dumps({"serverTime": 1700000000000, "symbols": []})
            expected_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

            response_id = db.insert_raw_api_response(
                endpoint="/api/v3/exchangeInfo",
                response_json=payload,
                fetched_at=1700000000000,
                params_json=json.dumps({"symbol": "BTCUSDT"}),
            )
            self.assertIsInstance(response_id, int)
            self.assertGreater(response_id, 0)

            conn = db.connection
            row = conn.execute(
                "SELECT * FROM raw_api_responses WHERE id = ?;",
                (response_id,),
            ).fetchone()

            self.assertEqual(row["endpoint"], "/api/v3/exchangeInfo")
            self.assertEqual(row["response_json"], payload)
            self.assertEqual(row["content_hash"], expected_hash)
            self.assertEqual(row["fetched_at"], 1700000000000)
            self.assertIsNotNone(row["created_at"])

            db.close()


class TestIngestionRuns(unittest.TestCase):
    """Tests for ingestion run lifecycle."""

    def test_ingestion_run_lifecycle(self):
        """Verify create → update → complete lifecycle."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)

            # Create
            run_id = db.create_ingestion_run(
                run_type="kline_backfill",
                started_at=1700000000000,
            )
            self.assertIsInstance(run_id, int)
            self.assertGreater(run_id, 0)

            conn = db.connection
            row = conn.execute(
                "SELECT * FROM ingestion_runs WHERE id = ?;", (run_id,)
            ).fetchone()
            self.assertEqual(row["run_type"], "kline_backfill")
            self.assertEqual(row["status"], "started")
            self.assertEqual(row["started_at"], 1700000000000)
            self.assertEqual(row["records_fetched"], 0)
            self.assertEqual(row["records_stored"], 0)

            # Update with progress
            db.update_ingestion_run(
                run_id=run_id,
                status="completed",
                completed_at=1700001000000,
                records_fetched=500,
                records_stored=500,
            )

            row = conn.execute(
                "SELECT * FROM ingestion_runs WHERE id = ?;", (run_id,)
            ).fetchone()
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["completed_at"], 1700001000000)
            self.assertEqual(row["records_fetched"], 500)
            self.assertEqual(row["records_stored"], 500)
            self.assertIsNone(row["error_message"])

            db.close()

    def test_ingestion_run_failure(self):
        """Verify failed ingestion run records error message."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)

            run_id = db.create_ingestion_run(
                run_type="exchange_info_sync",
                started_at=1700000000000,
            )

            db.update_ingestion_run(
                run_id=run_id,
                status="failed",
                completed_at=1700000100000,
                error_message="Connection timeout after 10s",
            )

            conn = db.connection
            row = conn.execute(
                "SELECT * FROM ingestion_runs WHERE id = ?;", (run_id,)
            ).fetchone()
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["error_message"], "Connection timeout after 10s")

            db.close()


class TestKlineRepository(unittest.TestCase):
    """Tests for kline insert, upsert, and query operations."""

    def test_kline_insert(self):
        """Verify single kline insert stores all columns correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)
            _insert_test_kline(db)

            results = db.query_klines_range("BTCUSDT", "1h", 1700000000000, 1700010000000)
            self.assertEqual(len(results), 1)

            k = results[0]
            self.assertEqual(k["symbol"], "BTCUSDT")
            self.assertEqual(k["interval"], "1h")
            self.assertEqual(k["open_time"], 1700000000000)
            self.assertEqual(k["open_price"], Decimal("50000.00000001"))
            self.assertEqual(k["high_price"], Decimal("51000.99999999"))
            self.assertEqual(k["low_price"], Decimal("49000.00000001"))
            self.assertEqual(k["close_price"], Decimal("50500.12345678"))
            self.assertEqual(k["volume"], Decimal("123.45678900"))
            self.assertEqual(k["close_time"], 1700003599999)
            self.assertEqual(k["number_of_trades"], 15420)

            db.close()

    def test_kline_upsert(self):
        """Verify upsert updates an existing kline without error."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)
            _insert_test_kline(db, close_price=Decimal("50500.00"))

            # Upsert with updated close price (forming candle scenario)
            db.upsert_kline(
                symbol="BTCUSDT",
                interval="1h",
                open_time=1700000000000,
                open_price=Decimal("50000.00000001"),
                high_price=Decimal("52000.00000000"),
                low_price=Decimal("49000.00000001"),
                close_price=Decimal("51800.55555555"),
                volume=Decimal("200.00000000"),
                close_time=1700003599999,
                quote_asset_volume=Decimal("10000000.00000000"),
                number_of_trades=20000,
                taker_buy_base_volume=Decimal("100.00000000"),
                taker_buy_quote_volume=Decimal("5000000.00000000"),
            )

            results = db.query_klines_range("BTCUSDT", "1h", 1700000000000, 1700010000000)
            self.assertEqual(len(results), 1)

            k = results[0]
            self.assertEqual(k["close_price"], Decimal("51800.55555555"))
            self.assertEqual(k["high_price"], Decimal("52000.00000000"))
            self.assertEqual(k["volume"], Decimal("200.00000000"))
            self.assertEqual(k["number_of_trades"], 20000)

            db.close()

    def test_kline_range_query_asc(self):
        """Verify range query returns klines in open_time ASC order."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            # Insert 5 klines at 1-hour intervals
            base_time = 1700000000000
            for i in range(5):
                _insert_test_kline(
                    db,
                    open_time=base_time + i * 3600000,
                    close_price=Decimal(f"{50000 + i * 100}"),
                )

            results = db.query_klines_range(
                "BTCUSDT", "1h",
                base_time,
                base_time + 5 * 3600000,
            )
            self.assertEqual(len(results), 5)

            # Verify ascending order
            for i in range(len(results) - 1):
                self.assertLess(results[i]["open_time"], results[i + 1]["open_time"])

            db.close()

    def test_kline_range_query_exclusive_end(self):
        """Verify range query end_time is exclusive."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            base_time = 1700000000000
            _insert_test_kline(db, open_time=base_time)
            _insert_test_kline(db, open_time=base_time + 3600000)

            # end_time exactly at second kline's open_time should exclude it
            results = db.query_klines_range(
                "BTCUSDT", "1h",
                base_time,
                base_time + 3600000,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["open_time"], base_time)

            db.close()

    def test_kline_latest_query_desc(self):
        """Verify latest-N query returns klines in open_time DESC order."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            base_time = 1700000000000
            for i in range(10):
                _insert_test_kline(
                    db,
                    open_time=base_time + i * 3600000,
                    close_price=Decimal(f"{50000 + i * 100}"),
                )

            results = db.query_klines_latest("BTCUSDT", "1h", limit=3)
            self.assertEqual(len(results), 3)

            # Verify descending order (latest first)
            for i in range(len(results) - 1):
                self.assertGreater(results[i]["open_time"], results[i + 1]["open_time"])

            # The latest should be the last inserted
            self.assertEqual(
                results[0]["open_time"],
                base_time + 9 * 3600000,
            )

            db.close()

    def test_kline_cross_symbol_isolation(self):
        """Verify kline queries do not leak between symbols."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")
            _insert_test_symbol(db, "ETHUSDT")

            _insert_test_kline(db, symbol="BTCUSDT", open_time=1700000000000)
            _insert_test_kline(db, symbol="ETHUSDT", open_time=1700000000000)

            btc_results = db.query_klines_range(
                "BTCUSDT", "1h", 1700000000000, 1700010000000
            )
            self.assertEqual(len(btc_results), 1)
            self.assertEqual(btc_results[0]["symbol"], "BTCUSDT")

            eth_results = db.query_klines_range(
                "ETHUSDT", "1h", 1700000000000, 1700010000000
            )
            self.assertEqual(len(eth_results), 1)
            self.assertEqual(eth_results[0]["symbol"], "ETHUSDT")

            db.close()


class TestTickerSnapshots(unittest.TestCase):
    """Tests for ticker snapshot operations."""

    def test_ticker_snapshot_insertion_query(self):
        """Verify ticker snapshot insert and query."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            snap_id = db.insert_ticker_snapshot(
                symbol="BTCUSDT",
                price=Decimal("67890.12345678"),
                snapshot_at=1700000000000,
            )
            self.assertIsInstance(snap_id, int)
            self.assertGreater(snap_id, 0)

            # Insert a second snapshot at a later time
            snap_id_2 = db.insert_ticker_snapshot(
                symbol="BTCUSDT",
                price=Decimal("67900.00000001"),
                snapshot_at=1700001000000,
            )

            results = db.query_ticker_snapshots("BTCUSDT", limit=10)
            self.assertEqual(len(results), 2)

            # Newest first
            self.assertEqual(results[0]["snapshot_at"], 1700001000000)
            self.assertEqual(results[0]["price"], Decimal("67900.00000001"))
            self.assertIsInstance(results[0]["price"], Decimal)

            self.assertEqual(results[1]["snapshot_at"], 1700000000000)
            self.assertEqual(results[1]["price"], Decimal("67890.12345678"))

            db.close()


class TestTransactionBehavior(unittest.TestCase):
    """Tests for transaction rollback and data persistence."""

    def test_rollback_on_failed_transaction(self):
        """Verify partial writes are rolled back when a transaction fails."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            # Insert a valid kline
            _insert_test_kline(db, open_time=1700000000000)

            # Attempt to insert with invalid FK (non-existent symbol) inside
            # a manual transaction
            conn = db.connection
            try:
                conn.execute("BEGIN;")
                # This valid kline should be part of the transaction
                conn.execute(
                    """
                    INSERT INTO klines
                        (symbol, interval, open_time, open_price, high_price, low_price,
                         close_price, volume, close_time, quote_asset_volume,
                         number_of_trades, taker_buy_base_volume, taker_buy_quote_volume,
                         raw_response_id)
                    VALUES ('BTCUSDT', '1h', 1700003600000, '50000', '51000', '49000',
                            '50500', '100', 1700007199999, '5000000', 1000, '50', '2500000', NULL);
                    """,
                )
                # This should fail due to FK violation
                conn.execute(
                    """
                    INSERT INTO klines
                        (symbol, interval, open_time, open_price, high_price, low_price,
                         close_price, volume, close_time, quote_asset_volume,
                         number_of_trades, taker_buy_base_volume, taker_buy_quote_volume,
                         raw_response_id)
                    VALUES ('INVALIDSYM', '1h', 1700007200000, '50000', '51000', '49000',
                            '50500', '100', 1700010799999, '5000000', 1000, '50', '2500000', NULL);
                    """,
                )
                conn.execute("COMMIT;")
            except sqlite3.IntegrityError:
                conn.execute("ROLLBACK;")

            # The second kline (1700003600000) should NOT exist due to rollback
            results = db.query_klines_range("BTCUSDT", "1h", 1700000000000, 1700010000000)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["open_time"], 1700000000000)

            db.close()

    def test_reopen_database_preserves_data(self):
        """Verify data persists after closing and reopening the database."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "persist_test.db")

            # First session: create and populate
            db = Database(db_path)
            db.connect()
            db.initialize()
            _insert_test_symbol(db)
            _insert_test_kline(db)

            snap_id = db.insert_ticker_snapshot(
                symbol="BTCUSDT",
                price=Decimal("50000.50"),
                snapshot_at=1700000000000,
            )
            db.close()

            # Second session: reopen and verify
            db2 = Database(db_path)
            db2.connect()
            # Re-initialize is safe (IF NOT EXISTS)
            db2.initialize()

            version = db2.get_schema_version()
            self.assertEqual(version, SCHEMA_VERSION)

            results = db2.query_klines_range("BTCUSDT", "1h", 1700000000000, 1700010000000)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["close_price"], Decimal("50500.12345678"))

            snaps = db2.query_ticker_snapshots("BTCUSDT")
            self.assertEqual(len(snaps), 1)
            self.assertEqual(snaps[0]["price"], Decimal("50000.50"))

            db2.close()


class TestExplainQueryPlan(unittest.TestCase):
    """Verify kline queries use the PK index without temporary B-trees."""

    def test_explain_query_plan_no_temp_btree(self):
        """Verify both approved kline queries use PRIMARY KEY scan, no temp B-tree."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            try:
                conn = db.connection

                # Query 1: Range scan ASC
                plan_asc = conn.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT *
                    FROM klines
                    WHERE symbol = ?
                      AND interval = ?
                      AND open_time >= ?
                      AND open_time < ?
                    ORDER BY open_time ASC;
                    """,
                    ("BTCUSDT", "1h", 1700000000000, 1700010000000),
                ).fetchall()

                # sqlite3.Row from EXPLAIN QUERY PLAN has columns:
                # id, parent, notused, detail
                plan_asc_text = " ".join(row["detail"] for row in plan_asc)
                self.assertNotIn("TEMP B-TREE", plan_asc_text.upper())
                self.assertIn("PRIMARY KEY", plan_asc_text.upper())

                # Query 2: Latest-N DESC
                plan_desc = conn.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT *
                    FROM klines
                    WHERE symbol = ?
                      AND interval = ?
                      AND open_time <= ?
                    ORDER BY open_time DESC
                    LIMIT ?;
                    """,
                    ("BTCUSDT", "1h", 1700010000000, 5),
                ).fetchall()

                plan_desc_text = " ".join(row["detail"] for row in plan_desc)
                self.assertNotIn("TEMP B-TREE", plan_desc_text.upper())
                self.assertIn("PRIMARY KEY", plan_desc_text.upper())
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
