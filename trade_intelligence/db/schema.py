"""Approved Phase 1.3.1 database schema for Trade Intelligence.

All tables defined here follow the locked Phase 1.3.1 design:
- klines uses WITHOUT ROWID with PK (symbol, interval, open_time)
- Financial values stored as TEXT (Python Decimal)
- Timestamps stored as INTEGER (UTC epoch milliseconds)
- Foreign keys enforced via PRAGMA foreign_keys = ON
"""

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Schema version tracking for future migration safety
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Tracks API ingestion sessions with status lifecycle
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type        TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'started',
    started_at      INTEGER NOT NULL,
    completed_at    INTEGER,
    records_fetched INTEGER NOT NULL DEFAULT 0,
    records_stored  INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Immutable raw JSON archive with SHA-256 content hash
CREATE TABLE IF NOT EXISTS raw_api_responses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint        TEXT    NOT NULL,
    params_json     TEXT,
    response_json   TEXT    NOT NULL,
    content_hash    TEXT    NOT NULL,
    ingestion_run_id INTEGER,
    fetched_at      INTEGER NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (ingestion_run_id) REFERENCES ingestion_runs(id)
);

-- Current symbol metadata (latest state from exchangeInfo)
CREATE TABLE IF NOT EXISTS symbols (
    symbol                  TEXT PRIMARY KEY,
    base_asset              TEXT    NOT NULL,
    quote_asset             TEXT    NOT NULL,
    status                  TEXT    NOT NULL,
    is_spot_trading_allowed INTEGER NOT NULL DEFAULT 0,
    is_margin_trading_allowed INTEGER NOT NULL DEFAULT 0,
    base_asset_precision    INTEGER NOT NULL DEFAULT 8,
    quote_asset_precision   INTEGER NOT NULL DEFAULT 8,
    updated_at              INTEGER NOT NULL,
    created_at              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Append-only status/permission transition log
CREATE TABLE IF NOT EXISTS symbol_status_events (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol                  TEXT    NOT NULL,
    status                  TEXT    NOT NULL,
    is_spot_trading_allowed INTEGER NOT NULL,
    effective_at            INTEGER NOT NULL,
    raw_response_id         INTEGER,
    created_at              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol),
    FOREIGN KEY (raw_response_id) REFERENCES raw_api_responses(id)
);

-- OHLCV candlestick data
-- WITHOUT ROWID: stores rows directly in the PK B-tree for clustered access
-- PK (symbol, interval, open_time) enables bidirectional range scans without extra indexes
CREATE TABLE IF NOT EXISTS klines (
    symbol                      TEXT    NOT NULL,
    interval                    TEXT    NOT NULL,
    open_time                   INTEGER NOT NULL,
    open_price                  TEXT    NOT NULL,
    high_price                  TEXT    NOT NULL,
    low_price                   TEXT    NOT NULL,
    close_price                 TEXT    NOT NULL,
    volume                      TEXT    NOT NULL,
    close_time                  INTEGER NOT NULL,
    quote_asset_volume          TEXT    NOT NULL,
    number_of_trades            INTEGER NOT NULL,
    taker_buy_base_volume       TEXT    NOT NULL,
    taker_buy_quote_volume      TEXT    NOT NULL,
    raw_response_id             INTEGER,
    PRIMARY KEY (symbol, interval, open_time),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol),
    FOREIGN KEY (raw_response_id) REFERENCES raw_api_responses(id)
) WITHOUT ROWID;

-- Point-in-time price snapshots
CREATE TABLE IF NOT EXISTS ticker_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    price       TEXT    NOT NULL,
    snapshot_at INTEGER NOT NULL,
    raw_response_id INTEGER,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (symbol) REFERENCES symbols(symbol),
    FOREIGN KEY (raw_response_id) REFERENCES raw_api_responses(id)
);
"""
