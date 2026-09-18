# Trade Intelligence

Trade Intelligence is a private, Binance-only trading intelligence application. The current focus is building the market-data foundation.

## Project Scope & Boundaries

- **Exchange**: Binance only
- **Data Access**: Binance official public market-data APIs
- **Database**: SQLite (Phase 1.3.2 foundation implemented)
- **Deployment**: Local initially
- **Trading**: Completely disabled (no execution, order routing, or account management)
- **API Credentials**: Not required (read-only public market data only)
- **Out of Scope**: No AI/ML, trading recommendations, automated trading, technical analysis engines, candlestick detection, Binance Square integrations, WebSockets, Futures, or frontend.

## Current State: Phase 1.3.2 (Completed)

### Phase 1.2 — Binance Spot REST Client
- **Endpoints Supported**: Server time (`GET /api/v3/time`), exchange information (`GET /api/v3/exchangeInfo`), ticker prices (`GET /api/v3/ticker/price`), and kline/OHLCV data (`GET /api/v3/klines`).
- **Supported Kline Intervals**: `5m`, `15m`, `1h`, `4h`, and `1d` with strict interval validation.
- **Financial Precision**: Uses standard library `decimal.Decimal` for all normalized price and volume fields, avoiding IEEE-754 binary floating-point rounding errors.
- **Raw Data Preservation**: Preserves untouched Binance response payloads in `.raw` for debugging and future data archival.
- **Error Handling**: Custom exception hierarchy (`BinanceConnectionError`, `BinanceTimeoutError`, `BinanceHttpError`, `BinanceApiError`, `BinanceResponseError`) with safe parsing of non-integer error codes.
- **Client Validation**: Enforces kline limit boundaries (1 <= limit <= 1000).

### Phase 1.3.2 — SQLite Database Foundation
- **Schema**: 6 data tables (`ingestion_runs`, `raw_api_responses`, `symbols`, `symbol_status_events`, `klines`, `ticker_snapshots`) plus `schema_version` tracking.
- **Klines Architecture**: `WITHOUT ROWID` table with composite primary key `(symbol, interval, open_time)` for clustered bidirectional range scans.
- **Data Integrity**: Foreign keys enforced via `PRAGMA foreign_keys = ON`. Financial values stored as TEXT (Python `Decimal`). Timestamps stored as INTEGER (UTC epoch milliseconds).
- **Configurable Pragmas**: WAL journal mode, synchronous level, cache size, and busy timeout are centrally configurable.
- **Upsert Support**: Forming candles may be updated via `ON CONFLICT DO UPDATE` while finalized historical candles remain effectively immutable.
- **Repository Methods**: Insert/update symbols, status events, raw API responses, ingestion runs, klines (insert and upsert), ticker snapshots, and range/latest-N kline queries.

## Development Setup

### 1. Prerequisites
- Python 3.10+ (tested with Python 3.14)

### 2. Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Running Tests
Run the deterministic offline unit test suite:
```powershell
python -m unittest discover -s tests
```

Run the live Binance integration tests (opt-in, requires internet connectivity):
```powershell
$env:RUN_LIVE_TESTS="1"; python -m unittest tests/test_binance_integration.py
```

## Project Structure
```text
Trade-Intelligence/
├── .gitignore                          # Git exclusion rules
├── README.md                           # Project documentation
├── requirements.txt                    # Minimal dependencies for Binance public API
├── tests/
│   ├── __init__.py
│   ├── test_environment.py          # Environment & dependency sanity checks
│   ├── test_binance_client.py       # Mocked offline unit tests
│   ├── test_binance_integration.py  # Opt-in live Binance API integration tests
│   └── test_database.py            # SQLite database foundation tests
└── trade_intelligence/
    ├── __init__.py                  # Top-level exports
    ├── binance/
    │   ├── __init__.py              # Binance module interface
    │   ├── client.py                # BinanceRestClient implementation
    │   ├── enums.py                 # KlineInterval enumeration & validation
    │   ├── exceptions.py            # Custom exception hierarchy
    │   └── models.py                # Typed dataclasses (Decimal prices/volumes)
    └── db/
        ├── __init__.py              # Database module interface
        ├── database.py              # Database class (connection, schema, repository)
        ├── exceptions.py            # Database exception hierarchy
        └── schema.py                # Phase 1.3.1 approved DDL & version constant
```