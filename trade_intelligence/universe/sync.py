"""Authoritative Binance Spot market universe synchronizer for Trade Intelligence.

Implements Phase 1.4.1:
- Queries authoritative Binance Spot GET /api/v3/exchangeInfo.
- Archives immutable raw API responses with SHA-256 content hashes.
- Tracks ingestion runs with audit lifecycle status.
- Normalizes and batch upserts USDT trading pairs into the symbols table.
- Emits baseline and transition events to symbol_status_events without duplicates.
- Preserves historical, non-trading, and delisted symbols against survivorship bias.
- Enforces separate transaction boundaries for raw telemetry vs normalized state.
"""

from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Dict, List, Optional

from trade_intelligence.binance.client import BinanceRestClient
from trade_intelligence.db.database import Database

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncResult:
    """Summary metrics of a universe synchronization run."""

    run_id: int
    raw_response_id: int
    records_fetched: int
    records_stored: int
    new_symbols: int
    status_transitions: int
    delisted_symbols: int


class UniverseSyncer:
    """Synchronizes Binance Spot symbol universe into SQLite database."""

    def __init__(self, db: Database, client: BinanceRestClient) -> None:
        """Initialize universe synchronizer.

        Args:
            db: Initialized Database instance.
            client: Configured BinanceRestClient instance.
        """
        self._db = db
        self._client = client

    def sync_spot_universe(
        self,
        quote_asset: str = "USDT",
    ) -> SyncResult:
        """Synchronize Spot universe from Binance authoritative exchangeInfo.

        Data flow:
        1. Initialize ingestion_runs record with run_type='exchange_info_sync'.
        2. Fetch authoritative exchangeInfo via Binance REST client.
        3. Archive raw response into raw_api_responses with SHA-256 hash (Transaction 1).
        4. Query current symbol states from database.
        5. Filter and normalize symbols for the target quote asset.
        6. Diff incoming states against current states to detect:
           - New symbols (baseline status event).
           - Status or permission transitions (transition status event).
           - Delisted/missing symbols (delisting status event, without deleting row).
        7. Atomically upsert symbols and insert status events (Transaction 2).
        8. Update ingestion_runs to 'completed' with counts.

        Args:
            quote_asset: Target quote asset filter (default: 'USDT').

        Returns:
            SyncResult summary of the synchronization run.

        Raises:
            ValueError: If quote_asset is empty.
            BinanceClientError: If network or exchange communication fails.
            DatabaseError: If database persistence fails.
        """
        if not quote_asset or not isinstance(quote_asset, str) or not quote_asset.strip():
            raise ValueError("quote_asset must be a non-empty string.")
        target_qa = quote_asset.strip().upper()

        started_at = int(time.time() * 1000)
        run_id = self._db.create_ingestion_run(
            run_type="exchange_info_sync",
            started_at=started_at,
        )

        # ------------------------------------------------------------------
        # Transaction Boundary 1: Fetch & Raw Response Archival
        # ------------------------------------------------------------------
        try:
            exchange_info = self._client.get_exchange_info()
            fetched_at = int(time.time() * 1000)
            raw_json_str = json.dumps(exchange_info.raw, separators=(",", ":"))
            raw_response_id = self._db.insert_raw_api_response(
                endpoint="/api/v3/exchangeInfo",
                response_json=raw_json_str,
                fetched_at=fetched_at,
                params_json="{}",
                ingestion_run_id=run_id,
            )
        except Exception as exc:
            now_ms = int(time.time() * 1000)
            self._db.update_ingestion_run(
                run_id=run_id,
                status="failed",
                completed_at=now_ms,
                error_message=str(exc),
            )
            raise

        # ------------------------------------------------------------------
        # Transaction Boundary 2: Normalization, Diffing & State Mutation
        # ------------------------------------------------------------------
        try:
            server_time_ms = exchange_info.server_time_ms or fetched_at
            existing_symbols = self._db.get_symbols()
            existing_map = {row["symbol"]: row for row in existing_symbols}

            symbols_to_upsert: List[Dict[str, Any]] = []
            events_to_insert: List[Dict[str, Any]] = []

            new_symbols_count = 0
            status_transitions_count = 0
            delisted_symbols_count = 0

            incoming_matching: Dict[str, Dict[str, Any]] = {}

            # Process symbols from exchange_info
            for s_info in exchange_info.symbols.values():
                if s_info.quote_asset.strip().upper() != target_qa:
                    continue

                sym = s_info.symbol.strip().upper()
                norm_record = {
                    "symbol": sym,
                    "base_asset": s_info.base_asset.strip().upper(),
                    "quote_asset": s_info.quote_asset.strip().upper(),
                    "status": s_info.status.strip().upper(),
                    "is_spot_trading_allowed": bool(s_info.is_spot_trading_allowed),
                    "is_margin_trading_allowed": bool(s_info.is_margin_trading_allowed),
                    "base_asset_precision": int(s_info.base_asset_precision),
                    "quote_asset_precision": int(s_info.quote_asset_precision),
                    "updated_at": server_time_ms,
                }
                incoming_matching[sym] = norm_record

            # Detect new symbols and status transitions
            for sym, norm in incoming_matching.items():
                symbols_to_upsert.append(norm)
                if sym not in existing_map:
                    new_symbols_count += 1
                    events_to_insert.append({
                        "symbol": sym,
                        "status": norm["status"],
                        "is_spot_trading_allowed": norm["is_spot_trading_allowed"],
                        "effective_at": server_time_ms,
                        "raw_response_id": raw_response_id,
                    })
                else:
                    prev = existing_map[sym]
                    status_changed = (norm["status"] != prev["status"])
                    spot_changed = (norm["is_spot_trading_allowed"] != prev["is_spot_trading_allowed"])
                    if status_changed or spot_changed:
                        status_transitions_count += 1
                        events_to_insert.append({
                            "symbol": sym,
                            "status": norm["status"],
                            "is_spot_trading_allowed": norm["is_spot_trading_allowed"],
                            "effective_at": server_time_ms,
                            "raw_response_id": raw_response_id,
                        })

            # Detect delisted symbols (previously existing USDT symbols absent from exchangeInfo)
            for sym, prev in existing_map.items():
                if prev["quote_asset"] == target_qa:
                    if sym not in incoming_matching and prev["status"] != "DELISTED":
                        delisted_symbols_count += 1
                        symbols_to_upsert.append({
                            "symbol": sym,
                            "base_asset": prev["base_asset"],
                            "quote_asset": prev["quote_asset"],
                            "status": "DELISTED",
                            "is_spot_trading_allowed": False,
                            "is_margin_trading_allowed": prev["is_margin_trading_allowed"],
                            "base_asset_precision": prev["base_asset_precision"],
                            "quote_asset_precision": prev["quote_asset_precision"],
                            "updated_at": server_time_ms,
                        })
                        events_to_insert.append({
                            "symbol": sym,
                            "status": "DELISTED",
                            "is_spot_trading_allowed": False,
                            "effective_at": server_time_ms,
                            "raw_response_id": raw_response_id,
                        })

            # Atomic persistence of symbols and status events
            self._db.batch_upsert_universe_state(symbols_to_upsert, events_to_insert)

            # Finalize run status
            now_ms = int(time.time() * 1000)
            self._db.update_ingestion_run(
                run_id=run_id,
                status="completed",
                completed_at=now_ms,
                records_fetched=len(exchange_info.symbols),
                records_stored=len(symbols_to_upsert),
            )

            return SyncResult(
                run_id=run_id,
                raw_response_id=raw_response_id,
                records_fetched=len(exchange_info.symbols),
                records_stored=len(symbols_to_upsert),
                new_symbols=new_symbols_count,
                status_transitions=status_transitions_count,
                delisted_symbols=delisted_symbols_count,
            )
        except Exception as exc:
            now_ms = int(time.time() * 1000)
            self._db.update_ingestion_run(
                run_id=run_id,
                status="failed",
                completed_at=now_ms,
                error_message=str(exc),
            )
            raise
