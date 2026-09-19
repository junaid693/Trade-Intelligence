"""SQLite database foundation for Trade Intelligence.

Provides connection management, schema initialization, and repository methods
for all approved Phase 1.3.1 tables. Includes single-row and batch insert/upsert
operations. Financial values are stored as TEXT (Python Decimal) and timestamps
as INTEGER (UTC epoch milliseconds).
"""

import hashlib
import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typing import TypedDict, NotRequired
except ImportError:
    from typing_extensions import TypedDict, NotRequired

from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.db.exceptions import (
    DatabaseError,
    DatabaseInitError,
    DatabaseIntegrityError,
)
from trade_intelligence.db.schema import SCHEMA_SQL, SCHEMA_VERSION

# Default SQLite pragmas for operational performance.
# These are configurable via the Database constructor.
DEFAULT_PRAGMAS: Dict[str, Any] = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "cache_size": -64000,  # 64 MB (negative = KiB)
    "busy_timeout": 5000,  # 5 seconds
    "foreign_keys": 1,     # Always enforced
}


class KlineRow(TypedDict):
    """Structured input for a single kline row in batch operations.

    All fields match the ``klines`` table columns. Financial values must be
    ``Decimal`` instances; timestamps must be non-negative ``int`` epoch ms.
    """

    symbol: str
    interval: str
    open_time: int
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    close_time: int
    quote_asset_volume: Decimal
    number_of_trades: int
    taker_buy_base_volume: Decimal
    taker_buy_quote_volume: Decimal
    raw_response_id: NotRequired[Optional[int]]


def _rollback_quietly(conn: sqlite3.Connection) -> None:
    """Attempt rollback on connection without masking the original exception."""
    try:
        conn.rollback()
    except sqlite3.Error:
        pass


def _to_decimal_str(val: Any, name: str = "value") -> str:
    """Validate that val is a finite Decimal and convert to string.

    Guarantees that float values or non-finite values (NaN, Infinity) cannot
    contaminate the database.
    """
    if not isinstance(val, Decimal):
        raise TypeError(
            f"{name} must be a Decimal instance, got {type(val).__name__}: {val!r}"
        )
    if not val.is_finite():
        raise ValueError(f"{name} must be a finite Decimal, got: {val}")
    return str(val)


def _validate_timestamp(ts: Any, name: str = "timestamp") -> int:
    """Validate that ts is a non-negative integer epoch millisecond timestamp."""
    if isinstance(ts, bool) or not isinstance(ts, int):
        raise TypeError(
            f"{name} must be an integer epoch millisecond timestamp, got {type(ts).__name__}: {ts!r}"
        )
    if ts < 0:
        raise ValueError(f"{name} must be a non-negative integer, got: {ts}")
    return ts


def _validate_symbol(symbol: str) -> str:
    """Validate and normalize trading pair symbol to stripped uppercase."""
    if not isinstance(symbol, str):
        raise TypeError(
            f"Symbol must be a string, got {type(symbol).__name__}: {symbol!r}"
        )
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("Symbol cannot be empty.")
    return cleaned


def _validate_limit(limit: int, name: str = "limit") -> int:
    """Validate query limit parameter."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(
            f"{name} must be an integer, got {type(limit).__name__}: {limit!r}"
        )
    if limit < 1:
        raise ValueError(f"Invalid {name}: {limit}. Limit must be a positive integer.")
    return limit


def _validate_kline_row(row: Dict[str, Any], index: int) -> Tuple:
    """Validate a kline dict and return a parameter tuple for executemany.

    Applies the same validation as ``insert_kline`` / ``upsert_kline``:
    symbol normalization, interval enum validation, timestamp validation,
    and finite-Decimal enforcement for all financial fields.

    Args:
        row: Dict with KlineRow keys.
        index: Row index in the batch (for error messages).

    Returns:
        Tuple of validated parameters in column order.

    Raises:
        TypeError: If a value has the wrong type.
        ValueError: If a value is invalid (e.g. unsupported interval).
    """
    try:
        sym = _validate_symbol(row["symbol"])
        iv = KlineInterval.from_value(row["interval"]).value
        ot = _validate_timestamp(row["open_time"], "open_time")
        ct = _validate_timestamp(row["close_time"], "close_time")
        op = _to_decimal_str(row["open_price"], "open_price")
        hp = _to_decimal_str(row["high_price"], "high_price")
        lp = _to_decimal_str(row["low_price"], "low_price")
        cp = _to_decimal_str(row["close_price"], "close_price")
        vol = _to_decimal_str(row["volume"], "volume")
        qav = _to_decimal_str(row["quote_asset_volume"], "quote_asset_volume")
        tbv = _to_decimal_str(row["taker_buy_base_volume"], "taker_buy_base_volume")
        tqv = _to_decimal_str(row["taker_buy_quote_volume"], "taker_buy_quote_volume")
        nt = row["number_of_trades"]
        if isinstance(nt, bool) or not isinstance(nt, int):
            raise TypeError(
                f"number_of_trades must be an integer, got {type(nt).__name__}: {nt!r}"
            )
        rid = row.get("raw_response_id")
    except KeyError as exc:
        raise ValueError(
            f"Kline row at index {index} is missing required field: {exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"Kline row at index {index}: {exc}"
        ) from exc

    return (sym, iv, ot, op, hp, lp, cp, vol, ct, qav, nt, tbv, tqv, rid)


class Database:
    """SQLite database manager for Trade Intelligence.

    Manages connection lifecycle, schema initialization, and provides
    repository methods for all Phase 1.3.1 tables.

    Usage::

        db = Database("trade_intelligence.db")
        db.connect()
        db.initialize()
        # ... use repository methods ...
        db.close()

    Or as a context manager::

        with Database("trade_intelligence.db") as db:
            db.initialize()
            # ... use repository methods ...
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        pragmas: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file. Use ":memory:" for
                     in-memory databases (useful for testing).
            pragmas: Optional dict of SQLite PRAGMA settings to override
                     defaults. The ``foreign_keys`` pragma is always forced
                     to 1 regardless of caller input.
        """
        self._db_path = str(db_path)
        self._pragmas = dict(DEFAULT_PRAGMAS)
        if pragmas:
            self._pragmas.update(pragmas)
        # Foreign keys are non-negotiable
        self._pragmas["foreign_keys"] = 1
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open a connection to the SQLite database and apply pragmas.

        Raises:
            DatabaseError: If the connection cannot be established.
        """
        if self._conn is not None:
            return
        try:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._apply_pragmas()
        except sqlite3.Error as exc:
            self._conn = None
            raise DatabaseError(f"Failed to connect to database: {exc}") from exc

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            finally:
                self._conn = None

    def __enter__(self) -> "Database":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the active connection, raising if not connected.

        Raises:
            DatabaseError: If not connected.
        """
        if self._conn is None:
            raise DatabaseError("Database is not connected. Call connect() first.")
        return self._conn

    def _apply_pragmas(self) -> None:
        """Apply configured PRAGMA settings to the active connection."""
        conn = self.connection
        for pragma, value in self._pragmas.items():
            if not pragma.replace("_", "").isalnum():
                raise DatabaseError(f"Invalid PRAGMA name: '{pragma}'")
            conn.execute(f"PRAGMA {pragma} = {value};")

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create all tables and record the schema version.

        Safe to call multiple times (uses CREATE TABLE IF NOT EXISTS).

        Raises:
            DatabaseInitError: If schema creation fails or schema version is unsupported.
        """
        conn = self.connection
        try:
            existing = self.get_schema_version()
            if existing is not None and existing > SCHEMA_VERSION:
                raise DatabaseInitError(
                    f"Database schema version {existing} is newer than supported version {SCHEMA_VERSION}."
                )
            conn.executescript(SCHEMA_SQL)
            # Record schema version if not already recorded
            if existing is None:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?);",
                    (SCHEMA_VERSION,),
                )
                conn.commit()
        except DatabaseInitError:
            _rollback_quietly(conn)
            raise
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseInitError(
                f"Failed to initialize database schema: {exc}"
            ) from exc

    def get_schema_version(self) -> Optional[int]:
        """Return the current schema version, or None if not yet recorded.

        Raises:
            DatabaseError: If the query fails.
        """
        conn = self.connection
        try:
            row = conn.execute(
                "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1;"
            ).fetchone()
            return row["version"] if row else None
        except sqlite3.OperationalError:
            # Table does not exist yet
            return None
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Failed to read schema version: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Repository: symbols
    # ------------------------------------------------------------------

    def insert_symbol(
        self,
        symbol: str,
        base_asset: str,
        quote_asset: str,
        status: str,
        is_spot_trading_allowed: bool,
        is_margin_trading_allowed: bool,
        base_asset_precision: int,
        quote_asset_precision: int,
        updated_at: int,
    ) -> None:
        """Insert or update a symbol record.

        Uses INSERT ... ON CONFLICT DO UPDATE to perform an in-place update
        when the symbol already exists. This preserves foreign-key
        relationships from child tables (klines, symbol_status_events,
        ticker_snapshots) that reference symbols(symbol).

        Note: INSERT OR REPLACE must NOT be used here because it performs
        a DELETE + INSERT internally, which would violate FK constraints
        from child rows or, if CASCADE were enabled, silently destroy
        all associated data.

        Args:
            symbol: Trading pair symbol (e.g. 'BTCUSDT').
            base_asset: Base asset name (e.g. 'BTC').
            quote_asset: Quote asset name (e.g. 'USDT').
            status: Symbol status (e.g. 'TRADING', 'BREAK').
            is_spot_trading_allowed: Whether spot trading is enabled.
            is_margin_trading_allowed: Whether margin trading is enabled.
            base_asset_precision: Decimal precision for the base asset.
            quote_asset_precision: Decimal precision for the quote asset.
            updated_at: UTC epoch milliseconds when this state was observed.

        Raises:
            DatabaseIntegrityError: If a constraint violation occurs.
            DatabaseError: If the insert fails.
        """
        sym = _validate_symbol(symbol)
        ts = _validate_timestamp(updated_at, "updated_at")
        conn = self.connection
        try:
            conn.execute(
                """
                INSERT INTO symbols
                    (symbol, base_asset, quote_asset, status,
                     is_spot_trading_allowed, is_margin_trading_allowed,
                     base_asset_precision, quote_asset_precision, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol) DO UPDATE SET
                    base_asset              = excluded.base_asset,
                    quote_asset             = excluded.quote_asset,
                    status                  = excluded.status,
                    is_spot_trading_allowed = excluded.is_spot_trading_allowed,
                    is_margin_trading_allowed = excluded.is_margin_trading_allowed,
                    base_asset_precision    = excluded.base_asset_precision,
                    quote_asset_precision   = excluded.quote_asset_precision,
                    updated_at              = excluded.updated_at;
                """,
                (
                    sym,
                    base_asset,
                    quote_asset,
                    status,
                    int(is_spot_trading_allowed),
                    int(is_margin_trading_allowed),
                    base_asset_precision,
                    quote_asset_precision,
                    ts,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error inserting symbol '{sym}': {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to insert symbol '{sym}': {exc}"
            ) from exc

    def get_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single symbol record by symbol name.

        Args:
            symbol: Trading pair symbol (e.g. 'BTCUSDT').

        Returns:
            Dict of symbol attributes, or None if not found.

        Raises:
            DatabaseError: If the query fails.
        """
        sym = _validate_symbol(symbol)
        conn = self.connection
        try:
            row = conn.execute(
                """
                SELECT symbol, base_asset, quote_asset, status,
                       is_spot_trading_allowed, is_margin_trading_allowed,
                       base_asset_precision, quote_asset_precision,
                       updated_at, created_at
                FROM symbols
                WHERE symbol = ?;
                """,
                (sym,),
            ).fetchone()
            if row is None:
                return None
            return {
                "symbol": row["symbol"],
                "base_asset": row["base_asset"],
                "quote_asset": row["quote_asset"],
                "status": row["status"],
                "is_spot_trading_allowed": bool(row["is_spot_trading_allowed"]),
                "is_margin_trading_allowed": bool(row["is_margin_trading_allowed"]),
                "base_asset_precision": row["base_asset_precision"],
                "quote_asset_precision": row["quote_asset_precision"],
                "updated_at": row["updated_at"],
                "created_at": row["created_at"],
            }
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to get symbol '{sym}': {exc}") from exc

    def get_symbols(
        self,
        quote_asset: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve symbols, optionally filtered by quote asset.

        Args:
            quote_asset: Optional quote asset filter (e.g. 'USDT').

        Returns:
            List of symbol dicts ordered by symbol ascending.

        Raises:
            ValueError: If quote_asset is an empty string.
            DatabaseError: If the query fails.
        """
        conn = self.connection
        try:
            if quote_asset is not None:
                if not isinstance(quote_asset, str) or not quote_asset.strip():
                    raise ValueError("quote_asset filter must be a non-empty string.")
                qa = quote_asset.strip().upper()
                rows = conn.execute(
                    """
                    SELECT symbol, base_asset, quote_asset, status,
                           is_spot_trading_allowed, is_margin_trading_allowed,
                           base_asset_precision, quote_asset_precision,
                           updated_at, created_at
                    FROM symbols
                    WHERE quote_asset = ?
                    ORDER BY symbol ASC;
                    """,
                    (qa,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT symbol, base_asset, quote_asset, status,
                           is_spot_trading_allowed, is_margin_trading_allowed,
                           base_asset_precision, quote_asset_precision,
                           updated_at, created_at
                    FROM symbols
                    ORDER BY symbol ASC;
                    """,
                ).fetchall()
            return [
                {
                    "symbol": row["symbol"],
                    "base_asset": row["base_asset"],
                    "quote_asset": row["quote_asset"],
                    "status": row["status"],
                    "is_spot_trading_allowed": bool(row["is_spot_trading_allowed"]),
                    "is_margin_trading_allowed": bool(row["is_margin_trading_allowed"]),
                    "base_asset_precision": row["base_asset_precision"],
                    "quote_asset_precision": row["quote_asset_precision"],
                    "updated_at": row["updated_at"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        except (ValueError, TypeError):
            raise
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to get symbols: {exc}") from exc

    # ------------------------------------------------------------------
    # Repository: symbol_status_events
    # ------------------------------------------------------------------

    def insert_symbol_status_event(
        self,
        symbol: str,
        status: str,
        is_spot_trading_allowed: bool,
        effective_at: int,
        raw_response_id: Optional[int] = None,
    ) -> int:
        """Append a symbol status transition event.

        Args:
            symbol: Trading pair symbol. Must exist in ``symbols`` table.
            status: New status value.
            is_spot_trading_allowed: New spot-trading permission.
            effective_at: UTC epoch milliseconds when the transition was observed.
            raw_response_id: Optional FK to ``raw_api_responses.id``.

        Returns:
            The rowid of the inserted event.

        Raises:
            DatabaseIntegrityError: If the referenced symbol does not exist.
            DatabaseError: If the insert fails.
        """
        sym = _validate_symbol(symbol)
        ts = _validate_timestamp(effective_at, "effective_at")
        conn = self.connection
        try:
            cursor = conn.execute(
                """
                INSERT INTO symbol_status_events
                    (symbol, status, is_spot_trading_allowed, effective_at, raw_response_id)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    sym,
                    status,
                    int(is_spot_trading_allowed),
                    ts,
                    raw_response_id,
                ),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error inserting status event for '{sym}': {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to insert status event for '{sym}': {exc}"
            ) from exc

    def query_symbol_status_events(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query status transition events for a symbol, newest first.

        Args:
            symbol: Trading pair symbol.
            limit: Maximum number of events to return.

        Returns:
            List of event dicts ordered by effective_at DESC, id DESC.

        Raises:
            ValueError: If limit < 1.
            DatabaseError: If the query fails.
        """
        sym = _validate_symbol(symbol)
        lim = _validate_limit(limit)
        conn = self.connection
        try:
            rows = conn.execute(
                """
                SELECT id, symbol, status, is_spot_trading_allowed,
                       effective_at, raw_response_id, created_at
                FROM symbol_status_events
                WHERE symbol = ?
                ORDER BY effective_at DESC, id DESC
                LIMIT ?;
                """,
                (sym, lim),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "symbol": row["symbol"],
                    "status": row["status"],
                    "is_spot_trading_allowed": bool(row["is_spot_trading_allowed"]),
                    "effective_at": row["effective_at"],
                    "raw_response_id": row["raw_response_id"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Failed to query status events for '{sym}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Repository: raw_api_responses
    # ------------------------------------------------------------------

    def insert_raw_api_response(
        self,
        endpoint: str,
        response_json: str,
        fetched_at: int,
        params_json: Optional[str] = None,
        ingestion_run_id: Optional[int] = None,
    ) -> int:
        """Store an immutable raw API response with SHA-256 content hash.

        Args:
            endpoint: API endpoint path (e.g. '/api/v3/exchangeInfo').
            response_json: The full raw JSON response string.
            fetched_at: UTC epoch milliseconds when the response was fetched.
            params_json: Optional JSON-encoded request parameters.
            ingestion_run_id: Optional FK to ``ingestion_runs.id``.

        Returns:
            The rowid of the inserted record.

        Raises:
            DatabaseIntegrityError: If foreign key constraint is violated.
            DatabaseError: If the insert fails.
        """
        ts = _validate_timestamp(fetched_at, "fetched_at")
        content_hash = hashlib.sha256(response_json.encode("utf-8")).hexdigest()
        conn = self.connection
        try:
            cursor = conn.execute(
                """
                INSERT INTO raw_api_responses
                    (endpoint, params_json, response_json, content_hash,
                     ingestion_run_id, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    endpoint,
                    params_json,
                    response_json,
                    content_hash,
                    ingestion_run_id,
                    ts,
                ),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error inserting raw response for '{endpoint}': {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to insert raw response for '{endpoint}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Repository: ingestion_runs
    # ------------------------------------------------------------------

    def create_ingestion_run(
        self,
        run_type: str,
        started_at: int,
    ) -> int:
        """Create a new ingestion run record.

        Args:
            run_type: Type descriptor (e.g. 'kline_backfill', 'exchange_info_sync').
            started_at: UTC epoch milliseconds when the run started.

        Returns:
            The rowid of the created run.

        Raises:
            DatabaseError: If the insert fails.
        """
        ts = _validate_timestamp(started_at, "started_at")
        conn = self.connection
        try:
            cursor = conn.execute(
                """
                INSERT INTO ingestion_runs (run_type, status, started_at)
                VALUES (?, 'started', ?);
                """,
                (run_type, ts),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error creating ingestion run: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to create ingestion run: {exc}"
            ) from exc

    def update_ingestion_run(
        self,
        run_id: int,
        status: str,
        completed_at: Optional[int] = None,
        records_fetched: Optional[int] = None,
        records_stored: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update an existing ingestion run record.

        Args:
            run_id: The rowid of the ingestion run to update.
            status: New status (e.g. 'completed', 'failed').
            completed_at: Optional UTC epoch ms completion timestamp.
            records_fetched: Optional count of records fetched.
            records_stored: Optional count of records stored.
            error_message: Optional error message on failure.

        Raises:
            DatabaseError: If the update fails or the run_id does not exist.
        """
        if completed_at is not None:
            _validate_timestamp(completed_at, "completed_at")
        conn = self.connection
        fields: List[str] = ["status = ?"]
        params: List[Any] = [status]

        if completed_at is not None:
            fields.append("completed_at = ?")
            params.append(completed_at)
        if records_fetched is not None:
            fields.append("records_fetched = ?")
            params.append(records_fetched)
        if records_stored is not None:
            fields.append("records_stored = ?")
            params.append(records_stored)
        if error_message is not None:
            fields.append("error_message = ?")
            params.append(error_message)

        params.append(run_id)

        try:
            cursor = conn.execute(
                f"UPDATE ingestion_runs SET {', '.join(fields)} WHERE id = ?;",
                params,
            )
            if cursor.rowcount == 0:
                raise DatabaseError(f"Ingestion run with id {run_id} not found.")
            conn.commit()
        except DatabaseError:
            _rollback_quietly(conn)
            raise
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to update ingestion run {run_id}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Repository: klines
    # ------------------------------------------------------------------

    def insert_kline(
        self,
        symbol: str,
        interval: Union[str, KlineInterval],
        open_time: int,
        open_price: Decimal,
        high_price: Decimal,
        low_price: Decimal,
        close_price: Decimal,
        volume: Decimal,
        close_time: int,
        quote_asset_volume: Decimal,
        number_of_trades: int,
        taker_buy_base_volume: Decimal,
        taker_buy_quote_volume: Decimal,
        raw_response_id: Optional[int] = None,
    ) -> None:
        """Insert a single kline row. Raises on PK conflict.

        Financial Decimal values are validated as finite Decimals and stored as TEXT.

        Raises:
            DatabaseIntegrityError: If the kline already exists (PK conflict)
                                    or a FK constraint is violated.
            DatabaseError: If the insert fails.
        """
        sym = _validate_symbol(symbol)
        iv = KlineInterval.from_value(interval).value
        ot = _validate_timestamp(open_time, "open_time")
        ct = _validate_timestamp(close_time, "close_time")
        op = _to_decimal_str(open_price, "open_price")
        hp = _to_decimal_str(high_price, "high_price")
        lp = _to_decimal_str(low_price, "low_price")
        cp = _to_decimal_str(close_price, "close_price")
        vol = _to_decimal_str(volume, "volume")
        qav = _to_decimal_str(quote_asset_volume, "quote_asset_volume")
        tbv = _to_decimal_str(taker_buy_base_volume, "taker_buy_base_volume")
        tqv = _to_decimal_str(taker_buy_quote_volume, "taker_buy_quote_volume")

        conn = self.connection
        try:
            conn.execute(
                """
                INSERT INTO klines
                    (symbol, interval, open_time, open_price, high_price, low_price,
                     close_price, volume, close_time, quote_asset_volume,
                     number_of_trades, taker_buy_base_volume, taker_buy_quote_volume,
                     raw_response_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    sym, iv, ot,
                    op, hp, lp, cp, vol, ct,
                    qav, number_of_trades, tbv, tqv,
                    raw_response_id,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error inserting kline ({sym}, {iv}, {ot}): {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to insert kline ({sym}, {iv}, {ot}): {exc}"
            ) from exc

    def upsert_kline(
        self,
        symbol: str,
        interval: Union[str, KlineInterval],
        open_time: int,
        open_price: Decimal,
        high_price: Decimal,
        low_price: Decimal,
        close_price: Decimal,
        volume: Decimal,
        close_time: int,
        quote_asset_volume: Decimal,
        number_of_trades: int,
        taker_buy_base_volume: Decimal,
        taker_buy_quote_volume: Decimal,
        raw_response_id: Optional[int] = None,
    ) -> None:
        """Insert or update a kline row via ON CONFLICT DO UPDATE.

        Used for forming candles that may receive updated data, and for
        deliberate re-ingestion of finalized candles.

        Raises:
            DatabaseIntegrityError: If a FK constraint is violated.
            DatabaseError: If the upsert fails.
        """
        sym = _validate_symbol(symbol)
        iv = KlineInterval.from_value(interval).value
        ot = _validate_timestamp(open_time, "open_time")
        ct = _validate_timestamp(close_time, "close_time")
        op = _to_decimal_str(open_price, "open_price")
        hp = _to_decimal_str(high_price, "high_price")
        lp = _to_decimal_str(low_price, "low_price")
        cp = _to_decimal_str(close_price, "close_price")
        vol = _to_decimal_str(volume, "volume")
        qav = _to_decimal_str(quote_asset_volume, "quote_asset_volume")
        tbv = _to_decimal_str(taker_buy_base_volume, "taker_buy_base_volume")
        tqv = _to_decimal_str(taker_buy_quote_volume, "taker_buy_quote_volume")

        conn = self.connection
        try:
            conn.execute(
                """
                INSERT INTO klines
                    (symbol, interval, open_time, open_price, high_price, low_price,
                     close_price, volume, close_time, quote_asset_volume,
                     number_of_trades, taker_buy_base_volume, taker_buy_quote_volume,
                     raw_response_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, interval, open_time) DO UPDATE SET
                    open_price              = excluded.open_price,
                    high_price              = excluded.high_price,
                    low_price               = excluded.low_price,
                    close_price             = excluded.close_price,
                    volume                  = excluded.volume,
                    close_time              = excluded.close_time,
                    quote_asset_volume      = excluded.quote_asset_volume,
                    number_of_trades        = excluded.number_of_trades,
                    taker_buy_base_volume   = excluded.taker_buy_base_volume,
                    taker_buy_quote_volume  = excluded.taker_buy_quote_volume,
                    raw_response_id         = excluded.raw_response_id;
                """,
                (
                    sym, iv, ot,
                    op, hp, lp, cp, vol, ct,
                    qav, number_of_trades, tbv, tqv,
                    raw_response_id,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error upserting kline ({sym}, {iv}, {ot}): {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to upsert kline ({sym}, {iv}, {ot}): {exc}"
            ) from exc

    def query_klines_range(
        self,
        symbol: str,
        interval: Union[str, KlineInterval],
        start_time: int,
        end_time: int,
    ) -> List[Dict[str, Any]]:
        """Query klines within a time range, ordered by open_time ASC.

        Args:
            symbol: Trading pair symbol.
            interval: Kline interval (e.g. '1h' or KlineInterval enum).
            start_time: Inclusive start UTC epoch ms.
            end_time: Exclusive end UTC epoch ms.

        Returns:
            List of kline dicts with Decimal financial values and int timestamps.

        Raises:
            DatabaseError: If the query fails.
        """
        sym = _validate_symbol(symbol)
        iv = KlineInterval.from_value(interval).value
        st = _validate_timestamp(start_time, "start_time")
        et = _validate_timestamp(end_time, "end_time")
        conn = self.connection
        try:
            rows = conn.execute(
                """
                SELECT symbol, interval, open_time, open_price, high_price,
                       low_price, close_price, volume, close_time,
                       quote_asset_volume, number_of_trades,
                       taker_buy_base_volume, taker_buy_quote_volume,
                       raw_response_id
                FROM klines
                WHERE symbol = ? AND interval = ?
                  AND open_time >= ? AND open_time < ?
                ORDER BY open_time ASC;
                """,
                (sym, iv, st, et),
            ).fetchall()
            return [self._row_to_kline_dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Failed to query klines range for {sym}/{iv}: {exc}"
            ) from exc

    def query_kline_open_times_range(
        self,
        symbol: str,
        interval: Union[str, KlineInterval],
        start_time: int,
        end_time: int,
    ) -> List[int]:
        """Query kline open_time values within a time range, ordered ASC.

        Lightweight alternative to :meth:`query_klines_range` that returns
        only integer open_time timestamps without allocating full kline dicts.
        Useful for continuity / coverage scanning over large ranges.

        Args:
            symbol: Trading pair symbol.
            interval: Kline interval (e.g. '1h' or KlineInterval enum).
            start_time: Inclusive start UTC epoch ms (open_time >= start_time).
            end_time: Inclusive end UTC epoch ms (open_time <= end_time).

        Returns:
            List of integer open_time timestamps ordered ascending.

        Raises:
            DatabaseError: If the query fails.
        """
        sym = _validate_symbol(symbol)
        iv = KlineInterval.from_value(interval).value
        st = _validate_timestamp(start_time, "start_time")
        et = _validate_timestamp(end_time, "end_time")
        conn = self.connection
        try:
            rows = conn.execute(
                """
                SELECT open_time
                FROM klines
                WHERE symbol = ? AND interval = ?
                  AND open_time >= ? AND open_time <= ?
                ORDER BY open_time ASC;
                """,
                (sym, iv, st, et),
            ).fetchall()
            return [int(row[0]) for row in rows]
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Failed to query kline open times for {sym}/{iv}: {exc}"
            ) from exc

    def query_klines_latest(
        self,
        symbol: str,
        interval: Union[str, KlineInterval],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Query the latest N klines, ordered by open_time DESC.

        Args:
            symbol: Trading pair symbol.
            interval: Kline interval (e.g. '1h' or KlineInterval enum).
            limit: Maximum number of klines to return (positive integer).

        Returns:
            List of kline dicts ordered by open_time DESC.

        Raises:
            ValueError: If limit < 1.
            DatabaseError: If the query fails.
        """
        sym = _validate_symbol(symbol)
        iv = KlineInterval.from_value(interval).value
        lim = _validate_limit(limit)
        conn = self.connection
        try:
            rows = conn.execute(
                """
                SELECT symbol, interval, open_time, open_price, high_price,
                       low_price, close_price, volume, close_time,
                       quote_asset_volume, number_of_trades,
                       taker_buy_base_volume, taker_buy_quote_volume,
                       raw_response_id
                FROM klines
                WHERE symbol = ? AND interval = ?
                ORDER BY open_time DESC
                LIMIT ?;
                """,
                (sym, iv, lim),
            ).fetchall()
            return [self._row_to_kline_dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Failed to query latest klines for {sym}/{iv}: {exc}"
            ) from exc

    def get_latest_kline_in_range(
        self,
        symbol: str,
        interval: Union[str, KlineInterval],
        start_time: int,
        end_time: int,
    ) -> Optional[Dict[str, Any]]:
        """Query the latest kline within a time range, ordered by open_time DESC LIMIT 1.

        Args:
            symbol: Trading pair symbol.
            interval: Kline interval (e.g. '1h' or KlineInterval enum).
            start_time: Inclusive start UTC epoch ms.
            end_time: Inclusive end UTC epoch ms.

        Returns:
            Kline dict if found, or None if no kline exists in the range.

        Raises:
            DatabaseError: If the query fails.
        """
        sym = _validate_symbol(symbol)
        iv = KlineInterval.from_value(interval).value
        st = _validate_timestamp(start_time, "start_time")
        et = _validate_timestamp(end_time, "end_time")
        conn = self.connection
        try:
            cursor = conn.execute(
                """
                SELECT symbol, interval, open_time, open_price, high_price,
                       low_price, close_price, volume, close_time,
                       quote_asset_volume, number_of_trades,
                       taker_buy_base_volume, taker_buy_quote_volume,
                       raw_response_id
                FROM klines
                WHERE symbol = ? AND interval = ?
                  AND open_time >= ? AND open_time <= ?
                ORDER BY open_time DESC
                LIMIT 1;
                """,
                (sym, iv, st, et),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_kline_dict(row)
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Failed to query latest kline in range for {sym}/{iv}: {exc}"
            ) from exc


    @staticmethod
    def _row_to_kline_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row from the klines table to a dict with Decimal values."""
        return {
            "symbol": row["symbol"],
            "interval": row["interval"],
            "open_time": row["open_time"],
            "open_price": Decimal(row["open_price"]),
            "high_price": Decimal(row["high_price"]),
            "low_price": Decimal(row["low_price"]),
            "close_price": Decimal(row["close_price"]),
            "volume": Decimal(row["volume"]),
            "close_time": row["close_time"],
            "quote_asset_volume": Decimal(row["quote_asset_volume"]),
            "number_of_trades": row["number_of_trades"],
            "taker_buy_base_volume": Decimal(row["taker_buy_base_volume"]),
            "taker_buy_quote_volume": Decimal(row["taker_buy_quote_volume"]),
            "raw_response_id": row["raw_response_id"],
        }

    # ------------------------------------------------------------------
    # Repository: ticker_snapshots
    # ------------------------------------------------------------------

    def insert_ticker_snapshot(
        self,
        symbol: str,
        price: Decimal,
        snapshot_at: int,
        raw_response_id: Optional[int] = None,
    ) -> int:
        """Store a point-in-time price snapshot.

        Args:
            symbol: Trading pair symbol. Must exist in ``symbols`` table.
            price: Price as finite Decimal, stored as TEXT.
            snapshot_at: UTC epoch milliseconds.
            raw_response_id: Optional FK to ``raw_api_responses.id``.

        Returns:
            The rowid of the inserted snapshot.

        Raises:
            DatabaseIntegrityError: If a FK constraint is violated.
            DatabaseError: If the insert fails.
        """
        sym = _validate_symbol(symbol)
        p = _to_decimal_str(price, "price")
        ts = _validate_timestamp(snapshot_at, "snapshot_at")
        conn = self.connection
        try:
            cursor = conn.execute(
                """
                INSERT INTO ticker_snapshots
                    (symbol, price, snapshot_at, raw_response_id)
                VALUES (?, ?, ?, ?);
                """,
                (sym, p, ts, raw_response_id),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error inserting ticker snapshot for '{sym}': {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to insert ticker snapshot for '{sym}': {exc}"
            ) from exc

    def query_ticker_snapshots(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query ticker snapshots for a symbol, newest first.

        Args:
            symbol: Trading pair symbol.
            limit: Maximum number of snapshots to return (positive integer).

        Returns:
            List of snapshot dicts with Decimal price and int timestamp.

        Raises:
            ValueError: If limit < 1.
            DatabaseError: If the query fails.
        """
        sym = _validate_symbol(symbol)
        lim = _validate_limit(limit)
        conn = self.connection
        try:
            rows = conn.execute(
                """
                SELECT id, symbol, price, snapshot_at, raw_response_id, created_at
                FROM ticker_snapshots
                WHERE symbol = ?
                ORDER BY snapshot_at DESC
                LIMIT ?;
                """,
                (sym, lim),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "symbol": row["symbol"],
                    "price": Decimal(row["price"]),
                    "snapshot_at": row["snapshot_at"],
                    "raw_response_id": row["raw_response_id"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Failed to query ticker snapshots for '{sym}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_upsert_symbols(
        self,
        symbols: List[Dict[str, Any]],
    ) -> int:
        """Insert or update multiple symbol records in a single transaction.

        Uses the same ``INSERT ... ON CONFLICT (symbol) DO UPDATE`` semantics
        as :meth:`insert_symbol`. All rows are validated before any SQL is
        executed (fail-fast). On any failure the entire batch is rolled back.

        Args:
            symbols: List of dicts, each with the same keys as
                     :meth:`insert_symbol` parameters: ``symbol``,
                     ``base_asset``, ``quote_asset``, ``status``,
                     ``is_spot_trading_allowed``, ``is_margin_trading_allowed``,
                     ``base_asset_precision``, ``quote_asset_precision``,
                     ``updated_at``.

        Returns:
            Count of rows processed.

        Raises:
            TypeError: If any row has an invalid type.
            ValueError: If any row has an invalid value.
            DatabaseIntegrityError: If a constraint violation occurs.
            DatabaseError: If the batch operation fails.
        """
        if not symbols:
            return 0

        # Validate all rows before touching the database
        validated: List[Tuple] = []
        for i, row in enumerate(symbols):
            try:
                sym = _validate_symbol(row["symbol"])
                ts = _validate_timestamp(row["updated_at"], "updated_at")
                validated.append((
                    sym,
                    row["base_asset"],
                    row["quote_asset"],
                    row["status"],
                    int(row["is_spot_trading_allowed"]),
                    int(row["is_margin_trading_allowed"]),
                    row["base_asset_precision"],
                    row["quote_asset_precision"],
                    ts,
                ))
            except KeyError as exc:
                raise ValueError(
                    f"Symbol row at index {i} is missing required field: {exc}"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"Symbol row at index {i}: {exc}"
                ) from exc

        conn = self.connection
        sql = """
            INSERT INTO symbols
                (symbol, base_asset, quote_asset, status,
                 is_spot_trading_allowed, is_margin_trading_allowed,
                 base_asset_precision, quote_asset_precision, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol) DO UPDATE SET
                base_asset              = excluded.base_asset,
                quote_asset             = excluded.quote_asset,
                status                  = excluded.status,
                is_spot_trading_allowed = excluded.is_spot_trading_allowed,
                is_margin_trading_allowed = excluded.is_margin_trading_allowed,
                base_asset_precision    = excluded.base_asset_precision,
                quote_asset_precision   = excluded.quote_asset_precision,
                updated_at              = excluded.updated_at;
        """
        try:
            conn.execute("BEGIN")
            conn.executemany(sql, validated)
            conn.execute("COMMIT")
            return len(validated)
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error in batch symbol upsert: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to batch upsert symbols: {exc}"
            ) from exc

    def batch_insert_symbol_status_events(
        self,
        events: List[Dict[str, Any]],
    ) -> int:
        """Insert multiple symbol status transition events in a single transaction.

        Args:
            events: List of dicts, each with keys: ``symbol``, ``status``,
                    ``is_spot_trading_allowed``, ``effective_at``, and
                    optionally ``raw_response_id``.

        Returns:
            Count of events inserted.

        Raises:
            TypeError: If any event has an invalid type.
            ValueError: If any event has an invalid value.
            DatabaseIntegrityError: If a foreign key constraint is violated.
            DatabaseError: If the batch operation fails.
        """
        if not events:
            return 0

        validated: List[Tuple] = []
        for i, row in enumerate(events):
            try:
                sym = _validate_symbol(row["symbol"])
                status = str(row["status"]).strip().upper()
                if not status:
                    raise ValueError("Status cannot be empty.")
                is_spot = int(bool(row["is_spot_trading_allowed"]))
                eff_at = _validate_timestamp(row["effective_at"], "effective_at")
                raw_id = row.get("raw_response_id")
                if raw_id is not None and (isinstance(raw_id, bool) or not isinstance(raw_id, int)):
                    raise TypeError(
                        f"raw_response_id must be an integer, got {type(raw_id).__name__}: {raw_id!r}"
                    )
                validated.append((sym, status, is_spot, eff_at, raw_id))
            except KeyError as exc:
                raise ValueError(
                    f"Symbol status event row at index {i} is missing required field: {exc}"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"Symbol status event row at index {i}: {exc}"
                ) from exc

        conn = self.connection
        sql = """
            INSERT INTO symbol_status_events
                (symbol, status, is_spot_trading_allowed, effective_at, raw_response_id)
            VALUES (?, ?, ?, ?, ?);
        """
        try:
            conn.execute("BEGIN")
            conn.executemany(sql, validated)
            conn.execute("COMMIT")
            return len(validated)
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error in batch symbol status events insert: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to batch insert symbol status events: {exc}"
            ) from exc

    def batch_upsert_universe_state(
        self,
        symbols: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """Atomically upsert symbols and insert status events in a single transaction.

        Validates all inputs before any database modification. Guarantees that
        symbol updates and status events either both commit or both roll back.

        Args:
            symbols: List of symbol dicts for batch upsert.
            events: List of status event dicts for batch insert.

        Returns:
            Tuple of (symbols_upserted_count, events_inserted_count).

        Raises:
            TypeError: If any row has an invalid type.
            ValueError: If any row has an invalid value.
            DatabaseIntegrityError: If a constraint violation occurs.
            DatabaseError: If the transaction fails.
        """
        # Validate symbols
        validated_symbols: List[Tuple] = []
        for i, row in enumerate(symbols):
            try:
                sym = _validate_symbol(row["symbol"])
                ts = _validate_timestamp(row["updated_at"], "updated_at")
                validated_symbols.append((
                    sym,
                    row["base_asset"],
                    row["quote_asset"],
                    row["status"],
                    int(row["is_spot_trading_allowed"]),
                    int(row["is_margin_trading_allowed"]),
                    row["base_asset_precision"],
                    row["quote_asset_precision"],
                    ts,
                ))
            except KeyError as exc:
                raise ValueError(
                    f"Symbol row at index {i} is missing required field: {exc}"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"Symbol row at index {i}: {exc}") from exc

        # Validate events
        validated_events: List[Tuple] = []
        for i, row in enumerate(events):
            try:
                sym = _validate_symbol(row["symbol"])
                status = str(row["status"]).strip().upper()
                if not status:
                    raise ValueError("Status cannot be empty.")
                is_spot = int(bool(row["is_spot_trading_allowed"]))
                eff_at = _validate_timestamp(row["effective_at"], "effective_at")
                raw_id = row.get("raw_response_id")
                if raw_id is not None and (isinstance(raw_id, bool) or not isinstance(raw_id, int)):
                    raise TypeError(
                        f"raw_response_id must be an integer, got {type(raw_id).__name__}: {raw_id!r}"
                    )
                validated_events.append((sym, status, is_spot, eff_at, raw_id))
            except KeyError as exc:
                raise ValueError(
                    f"Symbol status event row at index {i} is missing required field: {exc}"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"Symbol status event row at index {i}: {exc}") from exc

        conn = self.connection
        symbols_sql = """
            INSERT INTO symbols
                (symbol, base_asset, quote_asset, status,
                 is_spot_trading_allowed, is_margin_trading_allowed,
                 base_asset_precision, quote_asset_precision, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol) DO UPDATE SET
                base_asset              = excluded.base_asset,
                quote_asset             = excluded.quote_asset,
                status                  = excluded.status,
                is_spot_trading_allowed = excluded.is_spot_trading_allowed,
                is_margin_trading_allowed = excluded.is_margin_trading_allowed,
                base_asset_precision    = excluded.base_asset_precision,
                quote_asset_precision   = excluded.quote_asset_precision,
                updated_at              = excluded.updated_at;
        """
        events_sql = """
            INSERT INTO symbol_status_events
                (symbol, status, is_spot_trading_allowed, effective_at, raw_response_id)
            VALUES (?, ?, ?, ?, ?);
        """
        try:
            conn.execute("BEGIN")
            if validated_symbols:
                conn.executemany(symbols_sql, validated_symbols)
            if validated_events:
                conn.executemany(events_sql, validated_events)
            conn.execute("COMMIT")
            return (len(validated_symbols), len(validated_events))
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error in batch universe state upsert: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to batch upsert universe state: {exc}"
            ) from exc

    def batch_upsert_klines(
        self,
        klines: List[Dict[str, Any]],
        *,
        chunk_size: int = 500,
    ) -> int:
        """Insert or update multiple kline rows in a single transaction.

        Uses the same ``INSERT ... ON CONFLICT (symbol, interval, open_time)
        DO UPDATE`` semantics as :meth:`upsert_kline`. All rows are validated
        before any SQL is executed (fail-fast). On any failure the entire
        batch is rolled back.

        The ``chunk_size`` parameter controls how many rows are sent per
        ``executemany`` call for memory efficiency. It does **not** affect
        transaction scope — the entire batch is always one atomic transaction.

        Args:
            klines: List of dicts matching :class:`KlineRow` keys.
            chunk_size: Number of rows per ``executemany`` call. Default 500.

        Returns:
            Count of rows processed.

        Raises:
            TypeError: If any row has an invalid type.
            ValueError: If any row has an invalid value or chunk_size < 1.
            DatabaseIntegrityError: If a FK/constraint violation occurs.
            DatabaseError: If the batch operation fails.
        """
        if not klines:
            return 0

        if chunk_size < 1:
            raise ValueError(
                f"chunk_size must be a positive integer, got: {chunk_size}"
            )

        # Validate all rows before touching the database
        validated: List[Tuple] = [
            _validate_kline_row(row, i) for i, row in enumerate(klines)
        ]

        conn = self.connection
        sql = """
            INSERT INTO klines
                (symbol, interval, open_time, open_price, high_price, low_price,
                 close_price, volume, close_time, quote_asset_volume,
                 number_of_trades, taker_buy_base_volume, taker_buy_quote_volume,
                 raw_response_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, interval, open_time) DO UPDATE SET
                open_price              = excluded.open_price,
                high_price              = excluded.high_price,
                low_price               = excluded.low_price,
                close_price             = excluded.close_price,
                volume                  = excluded.volume,
                close_time              = excluded.close_time,
                quote_asset_volume      = excluded.quote_asset_volume,
                number_of_trades        = excluded.number_of_trades,
                taker_buy_base_volume   = excluded.taker_buy_base_volume,
                taker_buy_quote_volume  = excluded.taker_buy_quote_volume,
                raw_response_id         = excluded.raw_response_id;
        """
        try:
            conn.execute("BEGIN")
            for start in range(0, len(validated), chunk_size):
                conn.executemany(sql, validated[start:start + chunk_size])
            conn.execute("COMMIT")
            return len(validated)
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error in batch kline upsert: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to batch upsert klines: {exc}"
            ) from exc

    def batch_insert_ticker_snapshots(
        self,
        snapshots: List[Dict[str, Any]],
    ) -> int:
        """Insert multiple ticker snapshots in a single transaction.

        Ticker snapshots are append-only (each gets a unique autoincrement id).
        All rows are validated before any SQL is executed. On any failure the
        entire batch is rolled back.

        Args:
            snapshots: List of dicts with keys: ``symbol``, ``price``,
                       ``snapshot_at``, and optionally ``raw_response_id``.

        Returns:
            Count of rows processed.

        Raises:
            TypeError: If any row has an invalid type.
            ValueError: If any row has an invalid value.
            DatabaseIntegrityError: If a FK constraint is violated.
            DatabaseError: If the batch operation fails.
        """
        if not snapshots:
            return 0

        validated: List[Tuple] = []
        for i, row in enumerate(snapshots):
            try:
                sym = _validate_symbol(row["symbol"])
                p = _to_decimal_str(row["price"], "price")
                ts = _validate_timestamp(row["snapshot_at"], "snapshot_at")
                rid = row.get("raw_response_id")
                validated.append((sym, p, ts, rid))
            except KeyError as exc:
                raise ValueError(
                    f"Ticker snapshot row at index {i} is missing required field: {exc}"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"Ticker snapshot row at index {i}: {exc}"
                ) from exc

        conn = self.connection
        sql = """
            INSERT INTO ticker_snapshots
                (symbol, price, snapshot_at, raw_response_id)
            VALUES (?, ?, ?, ?);
        """
        try:
            conn.execute("BEGIN")
            conn.executemany(sql, validated)
            conn.execute("COMMIT")
            return len(validated)
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error in batch ticker snapshot insert: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to batch insert ticker snapshots: {exc}"
            ) from exc

    def batch_insert_raw_api_responses(
        self,
        responses: List[Dict[str, Any]],
    ) -> List[int]:
        """Insert multiple raw API responses in a single transaction.

        Unlike other batch methods, this uses individual ``execute`` calls
        (not ``executemany``) because the caller needs the ``lastrowid`` of
        each inserted record for provenance linking (e.g. setting
        ``klines.raw_response_id``).

        SHA-256 content hashes are computed automatically for each response.

        Args:
            responses: List of dicts with keys: ``endpoint``,
                       ``response_json``, ``fetched_at``, and optionally
                       ``params_json`` and ``ingestion_run_id``.

        Returns:
            List of inserted rowids in the same order as the input.

        Raises:
            TypeError: If any row has an invalid type.
            ValueError: If any row has an invalid value.
            DatabaseIntegrityError: If a FK constraint is violated.
            DatabaseError: If the batch operation fails.
        """
        if not responses:
            return []

        # Validate all rows before touching the database
        validated: List[Tuple] = []
        for i, row in enumerate(responses):
            try:
                ts = _validate_timestamp(row["fetched_at"], "fetched_at")
                response_json = row["response_json"]
                content_hash = hashlib.sha256(
                    response_json.encode("utf-8")
                ).hexdigest()
                validated.append((
                    row["endpoint"],
                    row.get("params_json"),
                    response_json,
                    content_hash,
                    row.get("ingestion_run_id"),
                    ts,
                ))
            except KeyError as exc:
                raise ValueError(
                    f"Raw response row at index {i} is missing required field: {exc}"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"Raw response row at index {i}: {exc}"
                ) from exc

        conn = self.connection
        sql = """
            INSERT INTO raw_api_responses
                (endpoint, params_json, response_json, content_hash,
                 ingestion_run_id, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?);
        """
        ids: List[int] = []
        try:
            conn.execute("BEGIN")
            for params in validated:
                cursor = conn.execute(sql, params)
                ids.append(cursor.lastrowid)  # type: ignore[arg-type]
            conn.execute("COMMIT")
            return ids
        except sqlite3.IntegrityError as exc:
            _rollback_quietly(conn)
            raise DatabaseIntegrityError(
                f"Integrity error in batch raw response insert: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            _rollback_quietly(conn)
            raise DatabaseError(
                f"Failed to batch insert raw responses: {exc}"
            ) from exc
