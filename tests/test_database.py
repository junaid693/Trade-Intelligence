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

from trade_intelligence.binance.enums import KlineInterval
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

    def test_get_latest_kline_in_range(self):
        """Verify get_latest_kline_in_range returns the newest candle in bounds."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            base_time = 1700000000000
            for i in range(5):
                _insert_test_kline(
                    db,
                    open_time=base_time + i * 3600000,
                    close_price=Decimal(f"{50000 + i * 100}"),
                )

            # Within range covering all 5
            latest = db.get_latest_kline_in_range(
                "BTCUSDT", "1h",
                base_time,
                base_time + 10 * 3600000,
            )
            self.assertIsNotNone(latest)
            self.assertEqual(latest["open_time"], base_time + 4 * 3600000)

            # Within subrange [base_time, base_time + 2h]
            latest_sub = db.get_latest_kline_in_range(
                "BTCUSDT", "1h",
                base_time,
                base_time + 2 * 3600000,
            )
            self.assertIsNotNone(latest_sub)
            self.assertEqual(latest_sub["open_time"], base_time + 2 * 3600000)

            # Outside range (no candles)
            none_res = db.get_latest_kline_in_range(
                "BTCUSDT", "1h",
                base_time + 10 * 3600000,
                base_time + 20 * 3600000,
            )
            self.assertIsNone(none_res)

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


class TestDatabaseHardeningAndIntegrity(unittest.TestCase):
    """Regression and hardening tests for data integrity and error handling."""

    def test_rollback_on_integrity_error_cleans_transaction(self):
        """Verify that connection is not left in a dirty transaction after error."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            conn = db.connection
            self.assertFalse(conn.in_transaction)

            with self.assertRaises(DatabaseIntegrityError):
                db.insert_symbol_status_event("NONEXISTENT", "TRADING", True, 1000)

            # Transaction must be cleanly rolled back
            self.assertFalse(conn.in_transaction)
            db.close()

    def test_pragma_sql_injection_protection(self):
        """Verify invalid PRAGMA names containing SQL injection are rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(
                os.path.join(tmp, "test.db"),
                pragmas={"invalid; DROP TABLE klines; --": 1},
            )
            with self.assertRaises(DatabaseError) as ctx:
                db.connect()
            self.assertIn("Invalid PRAGMA name", str(ctx.exception))
            db.close()

    def test_unsupported_future_schema_version_rejected(self):
        """Verify opening a database with a future schema version raises DatabaseInitError."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "future.db")
            db = Database(db_path)
            db.connect()
            db.initialize()
            # Artificially insert a future schema version
            conn = db.connection
            conn.execute("INSERT INTO schema_version (version) VALUES (99);")
            conn.commit()
            db.close()

            # Reopening and initializing should fail
            db2 = Database(db_path)
            db2.connect()
            with self.assertRaises(DatabaseInitError) as ctx:
                db2.initialize()
            self.assertIn("newer than supported version", str(ctx.exception))
            db2.close()

    def test_update_nonexistent_ingestion_run_raises(self):
        """Verify updating a non-existent ingestion run raises DatabaseError."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            with self.assertRaises(DatabaseError) as ctx:
                db.update_ingestion_run(run_id=99999, status="failed")
            self.assertIn("not found", str(ctx.exception))
            db.close()

    def test_decimal_type_safety_rejects_float_and_nan(self):
        """Verify float, NaN, and Infinity values are rejected to prevent contamination."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            # Float should raise TypeError
            with self.assertRaises(TypeError):
                db.insert_kline(
                    symbol="BTCUSDT",
                    interval="1h",
                    open_time=1700000000000,
                    open_price=50000.0,  # float!
                    high_price=Decimal("51000"),
                    low_price=Decimal("49000"),
                    close_price=Decimal("50500"),
                    volume=Decimal("100"),
                    close_time=1700003599999,
                    quote_asset_volume=Decimal("5000000"),
                    number_of_trades=1000,
                    taker_buy_base_volume=Decimal("50"),
                    taker_buy_quote_volume=Decimal("2500000"),
                )

            # NaN should raise ValueError
            with self.assertRaises(ValueError):
                db.insert_ticker_snapshot("BTCUSDT", price=Decimal("NaN"), snapshot_at=1000)

            # Infinity should raise ValueError
            with self.assertRaises(ValueError):
                db.insert_ticker_snapshot("BTCUSDT", price=Decimal("Infinity"), snapshot_at=1000)

            # Float in ticker snapshot should raise TypeError
            with self.assertRaises(TypeError):
                db.insert_ticker_snapshot("BTCUSDT", price=50000.5, snapshot_at=1000)  # type: ignore[arg-type]

            db.close()

    def test_timestamp_validation(self):
        """Verify negative and non-integer timestamps are rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            # Negative timestamp raises ValueError
            with self.assertRaises(ValueError):
                _insert_test_kline(db, open_time=-1)

            # Float timestamp raises TypeError
            with self.assertRaises(TypeError):
                _insert_test_kline(db, open_time=1700000000000.5)  # type: ignore[arg-type]

            # Negative updated_at on symbol raises ValueError
            with self.assertRaises(ValueError):
                db.insert_symbol(
                    symbol="ETHUSDT",
                    base_asset="ETH",
                    quote_asset="USDT",
                    status="TRADING",
                    is_spot_trading_allowed=True,
                    is_margin_trading_allowed=False,
                    base_asset_precision=8,
                    quote_asset_precision=8,
                    updated_at=-500,
                )

            db.close()

    def test_symbol_validation_and_normalization(self):
        """Verify symbol whitespace/case normalization and empty string rejection."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)

            # Lowercase with whitespace should be normalized to stripped uppercase
            db.insert_symbol(
                symbol="  btcusdt  ",
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
            row = conn.execute("SELECT symbol FROM symbols;").fetchone()
            self.assertEqual(row["symbol"], "BTCUSDT")

            # Empty symbol raises ValueError
            with self.assertRaises(ValueError):
                db.insert_symbol(
                    symbol="   ",
                    base_asset="BTC",
                    quote_asset="USDT",
                    status="TRADING",
                    is_spot_trading_allowed=True,
                    is_margin_trading_allowed=False,
                    base_asset_precision=8,
                    quote_asset_precision=8,
                    updated_at=1700000000000,
                )

            # Non-string symbol raises TypeError
            with self.assertRaises(TypeError):
                db.insert_symbol(
                    symbol=123,  # type: ignore[arg-type]
                    base_asset="BTC",
                    quote_asset="USDT",
                    status="TRADING",
                    is_spot_trading_allowed=True,
                    is_margin_trading_allowed=False,
                    base_asset_precision=8,
                    quote_asset_precision=8,
                    updated_at=1700000000000,
                )

            db.close()

    def test_kline_interval_validation(self):
        """Verify kline interval validation against KlineInterval enum."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            # Invalid interval raises ValueError
            with self.assertRaises(ValueError):
                _insert_test_kline(db, interval="3m")

            # Valid KlineInterval enum succeeds
            _insert_test_kline(db, interval=KlineInterval.INTERVAL_4H, open_time=1700000000000)
            res = db.query_klines_latest("BTCUSDT", KlineInterval.INTERVAL_4H, limit=1)
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]["interval"], "4h")

            # Query with invalid interval raises ValueError
            with self.assertRaises(ValueError):
                db.query_klines_range("BTCUSDT", "invalid_interval", 1000, 2000)

            db.close()

    def test_limit_validation(self):
        """Verify query limits must be positive integers."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db)

            # limit=0 raises ValueError
            with self.assertRaises(ValueError):
                db.query_klines_latest("BTCUSDT", "1h", limit=0)

            # limit=-1 raises ValueError
            with self.assertRaises(ValueError):
                db.query_klines_latest("BTCUSDT", "1h", limit=-1)

            # limit as string raises TypeError
            with self.assertRaises(TypeError):
                db.query_klines_latest("BTCUSDT", "1h", limit="10")  # type: ignore[arg-type]

            # boolean limit raises TypeError
            with self.assertRaises(TypeError):
                db.query_klines_latest("BTCUSDT", "1h", limit=True)  # type: ignore[arg-type]

            # ticker snapshots limit validation
            with self.assertRaises(ValueError):
                db.query_ticker_snapshots("BTCUSDT", limit=0)

            with self.assertRaises(ValueError):
                db.query_ticker_snapshots("BTCUSDT", limit=-10)

            db.close()


def _generate_symbol_rows(
    count: int = 50,
    updated_at: int = 1700000000000,
) -> list:
    """Generate test symbol dicts for batch testing."""
    rows = []
    for i in range(count):
        base = f"T{i:03d}"
        rows.append({
            "symbol": f"{base}USDT",
            "base_asset": base,
            "quote_asset": "USDT",
            "status": "TRADING",
            "is_spot_trading_allowed": True,
            "is_margin_trading_allowed": False,
            "base_asset_precision": 8,
            "quote_asset_precision": 8,
            "updated_at": updated_at,
        })
    return rows


def _generate_kline_rows(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    count: int = 500,
    start_time: int = 1700000000000,
) -> list:
    """Generate realistic kline dicts for batch testing.

    Creates consecutive klines at the correct interval spacing with
    deterministic but varied OHLCV values.
    """
    # Interval durations in milliseconds
    interval_ms = {
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    step = interval_ms[interval]
    rows = []
    base_price = Decimal("50000")
    for i in range(count):
        ot = start_time + i * step
        # Deterministic variation based on index
        offset = Decimal(str(i % 100)) * Decimal("10")
        op = base_price + offset
        hp = op + Decimal("500")
        lp = op - Decimal("200")
        cp = op + Decimal("100")
        vol = Decimal("100") + Decimal(str(i % 50))
        qav = vol * cp
        nt = 1000 + i
        tbv = vol / Decimal("2")
        tqv = qav / Decimal("2")
        rows.append({
            "symbol": symbol,
            "interval": interval,
            "open_time": ot,
            "open_price": op,
            "high_price": hp,
            "low_price": lp,
            "close_price": cp,
            "volume": vol,
            "close_time": ot + step - 1,
            "quote_asset_volume": qav,
            "number_of_trades": nt,
            "taker_buy_base_volume": tbv,
            "taker_buy_quote_volume": tqv,
        })
    return rows


class TestBatchOperations(unittest.TestCase):
    """Tests for Phase 1.3.3 batch insert/upsert operations."""

    # ------------------------------------------------------------------
    # batch_upsert_symbols
    # ------------------------------------------------------------------

    def test_batch_upsert_symbols_inserts_new(self):
        """Insert 50 new symbols in one call; verify all 50 exist."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            rows = _generate_symbol_rows(50)
            count = db.batch_upsert_symbols(rows)

            self.assertEqual(count, 50)
            # Spot-check first and last
            r = db.connection.execute(
                "SELECT COUNT(*) FROM symbols"
            ).fetchone()[0]
            self.assertEqual(r, 50)
            r0 = db.connection.execute(
                "SELECT status FROM symbols WHERE symbol = ?",
                ("T000USDT",),
            ).fetchone()
            self.assertEqual(r0["status"], "TRADING")
            db.close()

    def test_batch_upsert_symbols_updates_existing(self):
        """Insert 10, then batch-upsert the same 10 with changed status."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            rows = _generate_symbol_rows(10)
            db.batch_upsert_symbols(rows)

            # Update status
            for row in rows:
                row["status"] = "BREAK"
                row["updated_at"] = 1700000001000
            count = db.batch_upsert_symbols(rows)

            self.assertEqual(count, 10)
            row = db.connection.execute(
                "SELECT status FROM symbols WHERE symbol = ?",
                ("T005USDT",),
            ).fetchone()
            self.assertEqual(row["status"], "BREAK")
            db.close()

    def test_batch_upsert_symbols_preserves_child_rows(self):
        """Batch-upserting a symbol must not destroy child kline rows."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")
            _insert_test_kline(db, "BTCUSDT")

            # Upsert the same symbol via batch
            db.batch_upsert_symbols([{
                "symbol": "BTCUSDT",
                "base_asset": "BTC",
                "quote_asset": "USDT",
                "status": "BREAK",
                "is_spot_trading_allowed": False,
                "is_margin_trading_allowed": False,
                "base_asset_precision": 8,
                "quote_asset_precision": 8,
                "updated_at": 1700000001000,
            }])

            # Child kline must survive
            klines = db.query_klines_latest("BTCUSDT", "1h", limit=10)
            self.assertEqual(len(klines), 1)
            db.close()

    def test_batch_upsert_symbols_empty_list(self):
        """Calling with empty list returns 0 without error."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            result = db.batch_upsert_symbols([])
            self.assertEqual(result, 0)
            db.close()

    def test_batch_upsert_symbols_validation_failure_rolls_back(self):
        """A bad timestamp in one row rejects the entire batch."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            rows = _generate_symbol_rows(5)
            rows[2]["updated_at"] = -1  # Invalid
            with self.assertRaises(ValueError):
                db.batch_upsert_symbols(rows)
            # No rows should have been written
            count = db.connection.execute(
                "SELECT COUNT(*) FROM symbols"
            ).fetchone()[0]
            self.assertEqual(count, 0)
            db.close()

    # ------------------------------------------------------------------
    # batch_upsert_klines
    # ------------------------------------------------------------------

    def test_batch_upsert_klines_inserts_new(self):
        """Insert 500 klines; verify count and spot-check first/last."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")
            rows = _generate_kline_rows(count=500)
            count = db.batch_upsert_klines(rows)

            self.assertEqual(count, 500)
            db_count = db.connection.execute(
                "SELECT COUNT(*) FROM klines WHERE symbol = 'BTCUSDT'"
            ).fetchone()[0]
            self.assertEqual(db_count, 500)

            # Spot-check first kline
            first = db.connection.execute(
                "SELECT open_price FROM klines WHERE symbol='BTCUSDT' AND interval='1h' ORDER BY open_time ASC LIMIT 1"
            ).fetchone()
            self.assertEqual(Decimal(first["open_price"]), Decimal("50000"))
            db.close()

    def test_batch_upsert_klines_upserts_existing(self):
        """Insert 10, then batch-upsert same 10 with updated close_price."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")
            rows = _generate_kline_rows(count=10)
            db.batch_upsert_klines(rows)

            # Update close prices
            for row in rows:
                row["close_price"] = Decimal("99999.99")
            db.batch_upsert_klines(rows)

            # Verify updates
            result = db.query_klines_range(
                "BTCUSDT", "1h",
                start_time=1700000000000,
                end_time=1700000000000 + 10 * 3_600_000,
            )
            for k in result:
                self.assertEqual(k["close_price"], Decimal("99999.99"))
            db.close()

    def test_batch_upsert_klines_mixed_insert_update(self):
        """Batch of 20 where 10 are new and 10 are updates."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")

            # Insert first 10
            first_10 = _generate_kline_rows(count=10)
            db.batch_upsert_klines(first_10)

            # Prepare batch: first 10 updated + 10 new
            for row in first_10:
                row["close_price"] = Decimal("77777.77")
            next_10 = _generate_kline_rows(
                count=10,
                start_time=1700000000000 + 10 * 3_600_000,
            )
            mixed = first_10 + next_10
            count = db.batch_upsert_klines(mixed)

            self.assertEqual(count, 20)
            total = db.connection.execute(
                "SELECT COUNT(*) FROM klines"
            ).fetchone()[0]
            self.assertEqual(total, 20)

            # Verify updated rows
            updated = db.query_klines_range(
                "BTCUSDT", "1h",
                start_time=1700000000000,
                end_time=1700000000000 + 10 * 3_600_000,
            )
            for k in updated:
                self.assertEqual(k["close_price"], Decimal("77777.77"))
            db.close()

    def test_batch_upsert_klines_empty_list(self):
        """Calling with empty list returns 0 without error."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            result = db.batch_upsert_klines([])
            self.assertEqual(result, 0)
            db.close()

    def test_batch_upsert_klines_validation_rejects_float(self):
        """A float price in one row rejects the entire batch."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")
            rows = _generate_kline_rows(count=5)
            rows[3]["open_price"] = 50000.0  # float, not Decimal
            with self.assertRaises(TypeError):
                db.batch_upsert_klines(rows)
            # Nothing written
            count = db.connection.execute(
                "SELECT COUNT(*) FROM klines"
            ).fetchone()[0]
            self.assertEqual(count, 0)
            db.close()

    def test_batch_upsert_klines_validation_rejects_bad_interval(self):
        """An unsupported interval rejects the entire batch."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")
            rows = _generate_kline_rows(count=3)
            rows[1]["interval"] = "2h"  # unsupported
            with self.assertRaises(ValueError):
                db.batch_upsert_klines(rows)
            count = db.connection.execute(
                "SELECT COUNT(*) FROM klines"
            ).fetchone()[0]
            self.assertEqual(count, 0)
            db.close()

    def test_batch_upsert_klines_fk_violation_rolls_back(self):
        """Referencing a non-existent symbol raises DatabaseIntegrityError."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            # Note: no symbol inserted
            rows = _generate_kline_rows(count=5)
            with self.assertRaises(DatabaseIntegrityError):
                db.batch_upsert_klines(rows)
            count = db.connection.execute(
                "SELECT COUNT(*) FROM klines"
            ).fetchone()[0]
            self.assertEqual(count, 0)
            db.close()

    def test_batch_upsert_klines_atomicity(self):
        """100 klines where the last has FK violation → 0 rows written."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")
            rows = _generate_kline_rows(count=100)
            # Make the last row reference a non-existent symbol
            rows[-1]["symbol"] = "NOSUCHSYMBOL"
            with self.assertRaises(DatabaseIntegrityError):
                db.batch_upsert_klines(rows)
            count = db.connection.execute(
                "SELECT COUNT(*) FROM klines"
            ).fetchone()[0]
            self.assertEqual(count, 0)
            db.close()

    def test_batch_upsert_klines_idempotent(self):
        """Calling the same batch twice produces identical results."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")
            rows = _generate_kline_rows(count=20)
            db.batch_upsert_klines(rows)
            db.batch_upsert_klines(rows)

            count = db.connection.execute(
                "SELECT COUNT(*) FROM klines"
            ).fetchone()[0]
            self.assertEqual(count, 20)

            # Verify data is identical
            result = db.query_klines_range(
                "BTCUSDT", "1h",
                start_time=1700000000000,
                end_time=1700000000000 + 20 * 3_600_000,
            )
            self.assertEqual(len(result), 20)
            for k, row in zip(result, rows):
                self.assertEqual(k["open_price"], row["open_price"])
                self.assertEqual(k["close_price"], row["close_price"])
            db.close()

    def test_batch_upsert_klines_with_raw_response_id(self):
        """Klines linked to a raw_response_id preserve provenance."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")

            # Create a raw response first
            raw_id = db.insert_raw_api_response(
                endpoint="/api/v3/klines",
                response_json='[["test"]]',
                fetched_at=1700000000000,
            )

            rows = _generate_kline_rows(count=5)
            for row in rows:
                row["raw_response_id"] = raw_id
            db.batch_upsert_klines(rows)

            result = db.query_klines_range(
                "BTCUSDT", "1h",
                start_time=1700000000000,
                end_time=1700000000000 + 5 * 3_600_000,
            )
            for k in result:
                self.assertEqual(k["raw_response_id"], raw_id)
            db.close()

    # ------------------------------------------------------------------
    # batch_insert_ticker_snapshots
    # ------------------------------------------------------------------

    def test_batch_insert_ticker_snapshots(self):
        """Insert 100 snapshots; verify count and ordering."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")

            snapshots = [
                {
                    "symbol": "BTCUSDT",
                    "price": Decimal("50000") + Decimal(str(i)),
                    "snapshot_at": 1700000000000 + i * 1000,
                }
                for i in range(100)
            ]
            count = db.batch_insert_ticker_snapshots(snapshots)
            self.assertEqual(count, 100)

            result = db.query_ticker_snapshots("BTCUSDT", limit=100)
            self.assertEqual(len(result), 100)
            # Should be newest-first
            self.assertGreater(
                result[0]["snapshot_at"],
                result[-1]["snapshot_at"],
            )
            db.close()

    def test_batch_insert_ticker_snapshots_empty(self):
        """Calling with empty list returns 0 without error."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            result = db.batch_insert_ticker_snapshots([])
            self.assertEqual(result, 0)
            db.close()

    # ------------------------------------------------------------------
    # batch_insert_raw_api_responses
    # ------------------------------------------------------------------

    def test_batch_insert_raw_api_responses(self):
        """Insert 5 raw responses; verify returned IDs and content_hash."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            responses = [
                {
                    "endpoint": "/api/v3/klines",
                    "response_json": json.dumps({"batch": i}),
                    "fetched_at": 1700000000000 + i * 1000,
                }
                for i in range(5)
            ]
            ids = db.batch_insert_raw_api_responses(responses)

            self.assertEqual(len(ids), 5)
            # IDs should be sequential
            for i in range(1, len(ids)):
                self.assertEqual(ids[i], ids[i - 1] + 1)

            # Verify content_hash computed
            for i, rid in enumerate(ids):
                row = db.connection.execute(
                    "SELECT content_hash FROM raw_api_responses WHERE id = ?",
                    (rid,),
                ).fetchone()
                expected_hash = hashlib.sha256(
                    json.dumps({"batch": i}).encode("utf-8")
                ).hexdigest()
                self.assertEqual(row["content_hash"], expected_hash)
            db.close()

    def test_batch_insert_raw_api_responses_with_ingestion_run(self):
        """Link raw responses to an ingestion run; verify FK."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            run_id = db.create_ingestion_run(
                run_type="kline_backfill",
                started_at=1700000000000,
            )
            responses = [
                {
                    "endpoint": "/api/v3/klines",
                    "response_json": json.dumps({"run": i}),
                    "fetched_at": 1700000000000 + i * 1000,
                    "ingestion_run_id": run_id,
                }
                for i in range(3)
            ]
            ids = db.batch_insert_raw_api_responses(responses)
            self.assertEqual(len(ids), 3)

            # Verify FK linkage
            for rid in ids:
                row = db.connection.execute(
                    "SELECT ingestion_run_id FROM raw_api_responses WHERE id = ?",
                    (rid,),
                ).fetchone()
                self.assertEqual(row["ingestion_run_id"], run_id)
            db.close()

    # ------------------------------------------------------------------
    # Performance gates
    # ------------------------------------------------------------------

    def test_batch_performance_500_klines(self):
        """Insert 500 klines in under 1 second."""
        import time

        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")
            rows = _generate_kline_rows(count=500)

            start = time.perf_counter()
            db.batch_upsert_klines(rows)
            elapsed = time.perf_counter() - start

            self.assertLess(elapsed, 1.0, f"500 klines took {elapsed:.3f}s (> 1.0s)")
            db.close()

    def test_batch_performance_1000_klines(self):
        """Insert 1000 klines in under 2 seconds."""
        import time

        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")
            rows = _generate_kline_rows(count=1000)

            start = time.perf_counter()
            db.batch_upsert_klines(rows)
            elapsed = time.perf_counter() - start

            self.assertLess(elapsed, 2.0, f"1000 klines took {elapsed:.3f}s (> 2.0s)")
            db.close()

    def test_batch_insert_symbol_status_events(self):
        """Test batch insert of symbol status transition events."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")
            _insert_test_symbol(db, "ETHUSDT")

            events = [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "is_spot_trading_allowed": True,
                    "effective_at": 1700000000000,
                },
                {
                    "symbol": "ETHUSDT",
                    "status": "BREAK",
                    "is_spot_trading_allowed": False,
                    "effective_at": 1700000001000,
                },
            ]
            count = db.batch_insert_symbol_status_events(events)
            self.assertEqual(count, 2)

            btc_events = db.query_symbol_status_events("BTCUSDT")
            self.assertEqual(len(btc_events), 1)
            self.assertEqual(btc_events[0]["status"], "TRADING")
            self.assertTrue(btc_events[0]["is_spot_trading_allowed"])

            eth_events = db.query_symbol_status_events("ETHUSDT")
            self.assertEqual(len(eth_events), 1)
            self.assertEqual(eth_events[0]["status"], "BREAK")
            self.assertFalse(eth_events[0]["is_spot_trading_allowed"])
            db.close()

    def test_batch_insert_symbol_status_events_empty(self):
        """Batch insert empty list returns 0."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            self.assertEqual(db.batch_insert_symbol_status_events([]), 0)
            db.close()

    def test_batch_insert_symbol_status_events_fk_violation(self):
        """Batch insert with nonexistent symbol rolls back and raises."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            _insert_test_symbol(db, "BTCUSDT")
            events = [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "is_spot_trading_allowed": True,
                    "effective_at": 1700000000000,
                },
                {
                    "symbol": "NONEXISTENT",
                    "status": "TRADING",
                    "is_spot_trading_allowed": True,
                    "effective_at": 1700000001000,
                },
            ]
            with self.assertRaises(DatabaseIntegrityError):
                db.batch_insert_symbol_status_events(events)
            # Verify rollback: BTCUSDT has 0 events
            self.assertEqual(len(db.query_symbol_status_events("BTCUSDT")), 0)
            db.close()

    def test_get_symbol_and_get_symbols(self):
        """Test get_symbol and get_symbols queries."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_test_db(tmp)
            self.assertIsNone(db.get_symbol("BTCUSDT"))

            _insert_test_symbol(db, "BTCUSDT")
            db.insert_symbol(
                symbol="ETHBTC",
                base_asset="ETH",
                quote_asset="BTC",
                status="TRADING",
                is_spot_trading_allowed=True,
                is_margin_trading_allowed=False,
                base_asset_precision=8,
                quote_asset_precision=8,
                updated_at=1700000000000,
            )

            sym = db.get_symbol("btcusdt")
            self.assertIsNotNone(sym)
            self.assertEqual(sym["symbol"], "BTCUSDT")
            self.assertEqual(sym["base_asset"], "BTC")
            self.assertEqual(sym["quote_asset"], "USDT")

            all_syms = db.get_symbols()
            self.assertEqual(len(all_syms), 2)

            usdt_syms = db.get_symbols(quote_asset="USDT")
            self.assertEqual(len(usdt_syms), 1)
            self.assertEqual(usdt_syms[0]["symbol"], "BTCUSDT")

            btc_syms = db.get_symbols(quote_asset="BTC")
            self.assertEqual(len(btc_syms), 1)
            self.assertEqual(btc_syms[0]["symbol"], "ETHBTC")
            db.close()


if __name__ == "__main__":
    unittest.main()

