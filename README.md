# Trade Intelligence

Trade Intelligence is a private, Binance-only trading intelligence application. The current focus is building the market-data foundation.

## Project Scope & Boundaries

- **Exchange**: Binance only
- **Data Access**: Binance official public market-data APIs
- **Database**: SQLite (Phase 1.3.2 foundation & Phase 1.3.3 batch operations implemented)
- **Deployment**: Local initially
- **Trading**: Completely disabled (no execution, order routing, or account management)
- **API Credentials**: Not required (read-only public market data only)
- **Out of Scope**: No AI/ML, trading recommendations, automated trading, technical analysis engines, candlestick detection, Binance Square integrations, WebSockets, Futures, or frontend.

## Current State: Phase 1.4.3 (Completed)

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

### Phase 1.3.3 — Batch Insert / Upsert
- **Batch Operations**: High-throughput batch methods for `symbols`, `klines`, `ticker_snapshots`, and `raw_api_responses`.
- **Atomicity & Rollback**: Fail-fast validation with atomic `BEGIN` / `COMMIT` transactions and safe rollback on error.
- **FK Preservation**: Uses `INSERT ... ON CONFLICT DO UPDATE` semantics to maintain foreign-key integrity across updates.

### Phase 1.4.1 — Binance Spot Market Universe
- **Authoritative Synchronization**: `UniverseSyncer` syncs the complete Binance Spot USDT universe directly from `GET /api/v3/exchangeInfo`.
- **Dual Transaction Boundaries**: Raw API response archival committed in Boundary 1; normalized symbol state and status events committed atomically in Boundary 2.
- **Idempotent Status Ledger**: Tracks initial discovery, trading status changes, and Spot permission transitions in `symbol_status_events` with zero duplicate events on repeat runs.
- **Survivorship-Bias Protection**: Delisted or missing symbols are preserved in `symbols` as `DELISTED` (no deletions), maintaining referential integrity for all historical market data.

### Phase 1.4.2 — Historical Kline Downloader
- **Paginated Candlestick Acquisition**: Sequential historical pagination over Binance Spot `GET /api/v3/klines` up to 1,000 candles per page.
- **Strict Boundary Advancement**: Deterministic advancement by `next_start_time = C_last.close_time_ms + 1` with a stall assertion `next_start_time > current_start_time` preventing infinite pagination loops.
- **Forming Candle Exclusion**: Server-time clamping and filtering ensures no mutable, unfinalized candles enter historical storage. Safely terminates pagination if all returned candles are forming.
- **Zero-Auxiliary-State Tail Resume**: Resumes forward from the latest persisted candle directly from database state (`resume=True`), with decoupled `status='already_up_to_date'`.
- **Continuity & Gap Analysis**: `GapReport` accurately details missing intervals and candle counts across examined sequences without fabricating synthetic candles.
- **Adaptive Rate Limiting & Telemetry**: Dynamic inspection of `X-MBX-USED-WEIGHT-*` headers, conservative configurable pacing delay, and HTTP 429 backoff with `Retry-After`.
- **Provenance & Fault Isolation**: Commits canonical reconstructions of API payloads into `raw_api_responses` per page before batch-upserting normalized candles into `klines`, preserving full transaction boundaries.

### Phase 1.4.3 — Multi-Timeframe Historical Pipeline
- **Multi-Timeframe Historical Orchestration**: Sequential coordination across supported timeframes (`1d`, `4h`, `1h`, `15m`, `5m`) via `HistoricalDataOrchestrator`.
- **Symbol-First Sequential Execution**: Orchestration processes each symbol completely through all requested intervals before proceeding to the next symbol, ensuring bounded memory usage and predictable execution.
- **Coverage Scanning**: `CoverageScanner` calculates expected candle grids, aligns timestamp boundaries to timeframe multiples, and evaluates complete vs missing data.
- **Gap Classification**: Categorizes detected continuity gaps into `LEADING`, `INTERIOR`, `TRAILING`, and `FULL_RANGE` gap types.
- **Targeted Gap Repair**: `GapRepairer` constructs bounded `RepairSegment`s and executes targeted downloads through `HistoricalKlineDownloader` with zero duplicated retry logic.
- **Post-Repair Verification**: Re-scans repaired intervals to verify data continuity and classify any residual gaps.
- **Strict Request Validation**: `validate_pipeline_request` enforces fail-fast checking on `PipelineRequest` (rejecting boolean timestamps, truthy non-booleans, and invalid intervals/ranges).
- **Independent 3-Axis Statuses**: Decoupled `DownloadStatus`, `CoverageStatus`, and `RepairStatus` for transparent operational diagnostics.

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

#### Runtime / Production Dependencies
Install minimal runtime dependencies for the core application:
```powershell
pip install -r requirements.txt
```

#### Development & Testing Dependencies
Install test execution tools (`pytest`):
```powershell
pip install -r requirements-dev.txt
```

### 4. Running Tests

The full test suite relies on `pytest` for test discovery, fixtures, and parameterized execution across both unit and pipeline test suites.

Run the complete deterministic offline test suite:
```powershell
pytest
```
or via Python module runner:
```powershell
python -m pytest
```

Run the live Binance integration tests (opt-in, requires internet connectivity):
```powershell
$env:RUN_LIVE_TESTS="1"; pytest
```
or on Linux/macOS:
```bash
RUN_LIVE_TESTS="1" pytest
```

Standard library `unittest` may also be used to execute individual unittest-compatible modules:
```powershell
python -m unittest tests/test_database.py
```
*(Note: Full discovery of all 237 tests requires `pytest` because Phase 1.4.3 test modules use pytest features.)*

## Project Structure
```text
Trade-Intelligence/
├── .gitignore                          # Git exclusion rules
├── README.md                           # Project documentation
├── requirements.txt                    # Minimal dependencies for Binance public API
├── requirements-dev.txt                # Development and test dependencies (pytest)
├── tests/
│   ├── __init__.py
│   ├── test_environment.py          # Environment & dependency sanity checks
│   ├── test_binance_client.py       # Mocked offline unit tests
│   ├── test_binance_integration.py  # Opt-in live Binance API integration tests
│   ├── test_database.py            # SQLite database foundation tests
│   ├── test_universe_sync.py       # Universe synchronization unit tests
│   ├── test_universe_sync_integration.py # Opt-in live universe sync integration tests
│   ├── test_kline_downloader.py    # Historical kline downloader unit tests
│   ├── test_kline_downloader_integration.py # Opt-in live kline download integration tests
│   ├── test_kline_coverage.py      # Kline coverage scanner & alignment tests
│   ├── test_kline_repair.py        # Gap repair planning & execution tests
│   ├── test_kline_orchestrator.py  # Multi-timeframe orchestrator unit tests
│   ├── test_kline_orchestrator_integration.py # Multi-timeframe pipeline integration tests
│   └── test_system_integration.py  # End-to-end full system integration tests
└── trade_intelligence/
    ├── __init__.py                  # Top-level exports
    ├── binance/
    │   ├── __init__.py              # Binance module interface
    │   ├── client.py                # BinanceRestClient implementation
    │   ├── enums.py                 # KlineInterval enumeration & validation
    │   ├── exceptions.py            # Custom exception hierarchy
    │   └── models.py                # Typed dataclasses (Decimal prices/volumes)
    ├── db/
    │   ├── __init__.py              # Database module interface
    │   ├── database.py              # Database class (connection, schema, repository)
    │   ├── exceptions.py            # Database exception hierarchy
    │   └── schema.py                # Phase 1.3.1 approved DDL & version constant
    ├── klines/
    │   ├── __init__.py              # Klines module interface
    │   ├── downloader.py            # HistoricalKlineDownloader implementation
    │   ├── gap_detector.py          # Chronological gap detection & validation
    │   ├── types.py                 # DownloadResult, GapReport, KlineGap dataclasses
    │   └── pipeline/
    │       ├── __init__.py          # Pipeline module interface
    │       ├── coverage.py          # CoverageScanner & timestamp alignment
    │       ├── orchestrator.py      # HistoricalDataOrchestrator implementation
    │       ├── repair.py            # GapRepairer planning & execution
    │       └── types.py             # Pipeline request/result dataclasses & enums
    └── universe/
        ├── __init__.py              # Universe module interface
        └── sync.py                  # UniverseSyncer implementation & SyncResult
```