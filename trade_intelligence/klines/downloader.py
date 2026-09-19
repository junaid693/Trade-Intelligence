"""Historical Binance Spot kline downloader with pagination, gap detection, and provenance."""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

from trade_intelligence.binance.client import BinanceRestClient
from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.binance.exceptions import (
    BinanceConnectionError,
    BinanceHttpError,
    BinanceTimeoutError,
)
from trade_intelligence.binance.models import Kline
from trade_intelligence.db.database import Database, KlineRow
from trade_intelligence.klines.gap_detector import (
    detect_kline_gaps,
    interval_to_milliseconds,
)
from trade_intelligence.klines.types import DownloadResult, GapReport

logger = logging.getLogger(__name__)


class HistoricalKlineDownloader:
    """Acquires and persists historical candlestick data from Binance Spot REST API."""

    def __init__(
        self,
        db: Database,
        client: BinanceRestClient,
        request_delay_ms: float = 200.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize historical kline downloader.

        Args:
            db: Initialized Database instance.
            client: Configured BinanceRestClient instance.
            request_delay_ms: Conservative pacing delay between paginated requests (ms).
            max_retries: Maximum retries for transient errors and HTTP 429 rate limits.
        """
        self._db = db
        self._client = client
        self._request_delay_ms = max(0.0, float(request_delay_ms))
        self._max_retries = max(0, int(max_retries))
        self._last_headers: Dict[str, str] = {}

        # Attach telemetry response hook to the client session
        self._attach_session_hook()

    def _attach_session_hook(self) -> None:
        """Attach response hook to the client's session to capture headers."""
        if hasattr(self._client, "_session") and self._client._session is not None:
            hooks = self._client._session.hooks.setdefault("response", [])
            if self._capture_response_headers not in hooks:
                hooks.append(self._capture_response_headers)

    def _capture_response_headers(self, response: Any, *args: Any, **kwargs: Any) -> None:
        """Capture lowercase response headers from the HTTP response."""
        if hasattr(response, "headers") and response.headers:
            self._last_headers = {k.lower(): str(v) for k, v in response.headers.items()}

    def _handle_pacing_and_telemetry(self, base_delay_ms: float) -> None:
        """Enforce conservative request pacing and adapt dynamically to telemetry."""
        delay_s = max(0.0, base_delay_ms / 1000.0)

        # Inspect observed weight telemetry from response headers
        used_weight: Optional[int] = None
        for k, v in self._last_headers.items():
            if "used-weight" in k:
                try:
                    val = int(v)
                    if used_weight is None or val > used_weight:
                        used_weight = val
                except (ValueError, TypeError):
                    pass

        if used_weight is not None:
            if used_weight > 1000:
                delay_s = max(delay_s, 1.0)
                logger.warning(
                    "Observed high Binance used-weight (%d). Throttling delay to %.1fs.",
                    used_weight,
                    delay_s,
                )
            elif used_weight > 700:
                delay_s = max(delay_s, 0.5)

        if delay_s > 0:
            time.sleep(delay_s)

    def _fetch_klines_page(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: int,
        end_time: int,
        limit: int = 1000,
    ) -> List[Kline]:
        """Fetch a single page of klines with retry logic for transient errors and HTTP 429."""
        retries = 0
        while True:
            try:
                return self._client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit,
                )
            except BinanceHttpError as exc:
                if exc.status_code == 429:
                    retries += 1
                    if retries > self._max_retries:
                        raise
                    retry_after_str = self._last_headers.get("retry-after")
                    retry_after_s = 2.0
                    if retry_after_str:
                        try:
                            retry_after_s = max(float(retry_after_str), 1.0)
                        except ValueError:
                            pass
                    logger.warning(
                        "Received HTTP 429 Rate Limit from Binance. Sleeping %.1fs (retry %d/%d).",
                        retry_after_s,
                        retries,
                        self._max_retries,
                    )
                    time.sleep(retry_after_s)
                    continue
                raise
            except (BinanceTimeoutError, BinanceConnectionError) as exc:
                retries += 1
                if retries > self._max_retries:
                    raise
                backoff_s = float(2 ** (retries - 1))
                logger.warning(
                    "Transient network error (%s). Backing off for %.1fs (retry %d/%d).",
                    exc,
                    backoff_s,
                    retries,
                    self._max_retries,
                )
                time.sleep(backoff_s)

    def download_historical_klines(
        self,
        symbol: str,
        interval: Union[str, KlineInterval],
        start_time: int,
        end_time: int,
        resume: bool = True,
    ) -> DownloadResult:
        """Download and persist historical klines for a symbol and interval.

        Data flow:
        1. Validate inputs and ensure symbol exists in symbols table (FK protection).
        2. Create an ingestion_runs record (run_type='kline_download').
        3. Query server time and clamp effective_end_time = min(end_time, server_time - 1).
        4. If resume=True, perform Tail Resume from latest persisted candle in range.
        5. Paginate sequentially with limit=1000:
           - Filter forming candles (close_time >= server_time).
           - Safely terminate if all returned candles are forming.
           - Store canonical raw payload in raw_api_responses.
           - Batch upsert normalized klines referencing raw_response_id.
           - Advance next_start_time = last_closed_kline.close_time + 1.
           - Enforce pagination advancement invariant (assert next_start_time > current_start_time).
           - Conservative pacing and telemetry check.
        6. Perform continuity analysis on examined sequence (GapReport).
        7. Update ingestion_runs to 'completed' and return DownloadResult.

        Args:
            symbol: Trading pair symbol (must exist in symbols table).
            interval: Kline interval (e.g. '1h' or KlineInterval enum).
            start_time: Inclusive start UTC epoch ms.
            end_time: Inclusive end UTC epoch ms.
            resume: If True, resume from the latest persisted candle (Tail Resume).

        Returns:
            DownloadResult with operation metrics and GapReport.

        Raises:
            ValueError: If symbol, interval, or timestamps are invalid, or symbol is not in database.
            RuntimeError: If pagination stalls.
            BinanceClientError: If network or exchange error aborts the run.
            DatabaseError: If database persistence fails.
        """
        # Validate symbol
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        sym = symbol.strip().upper()

        # Validate symbol exists in symbols table to preserve foreign key constraints
        db_symbol = self._db.get_symbol(sym)
        if db_symbol is None:
            raise ValueError(
                f"Symbol '{sym}' not found in symbols table. Run universe sync first."
            )

        # Validate interval
        validated_interval = KlineInterval.from_value(interval)
        interval_ms = interval_to_milliseconds(validated_interval)

        # Validate timestamps
        if not isinstance(start_time, int) or start_time < 0:
            raise ValueError(f"start_time must be a non-negative integer, got: {start_time}")
        if not isinstance(end_time, int) or end_time < 0:
            raise ValueError(f"end_time must be a non-negative integer, got: {end_time}")
        if start_time > end_time:
            raise ValueError(f"start_time ({start_time}) cannot be greater than end_time ({end_time})")

        start_perf = time.perf_counter()
        now_ms = int(time.time() * 1000)

        # Initialize audit tracking
        run_id = self._db.create_ingestion_run(
            run_type="kline_download",
            started_at=now_ms,
        )

        records_fetched = 0
        records_stored = 0
        pages_fetched = 0
        all_stored_klines: List[Kline] = []

        try:
            # Query Binance server time for clamping
            st_obj = self._client.get_server_time()
            server_time_ms = getattr(st_obj, "server_time_ms", getattr(st_obj, "server_time", int(time.time() * 1000)))
            effective_end_time = min(end_time, server_time_ms - 1)

            # Align initial start time to interval boundary
            aligned_start_time = start_time - (start_time % interval_ms)
            current_start_time = aligned_start_time

            # Tail Resume check
            if resume:
                latest_kline = self._db.get_latest_kline_in_range(
                    symbol=sym,
                    interval=validated_interval,
                    start_time=aligned_start_time,
                    end_time=effective_end_time,
                )
                if latest_kline is not None:
                    resume_start_time = latest_kline["close_time"] + 1
                    if resume_start_time > effective_end_time:
                        logger.info(
                            "Tail Resume: %s %s is already up-to-date through %d.",
                            sym,
                            validated_interval.value,
                            effective_end_time,
                        )
                        self._db.update_ingestion_run(
                            run_id=run_id,
                            status="completed",
                            completed_at=int(time.time() * 1000),
                            records_fetched=0,
                            records_stored=0,
                        )
                        gap_report = detect_kline_gaps(
                            symbol=sym,
                            interval=validated_interval,
                            klines=[],
                            expected_start_time=None,
                            expected_end_time=None,
                        )
                        return DownloadResult(
                            symbol=sym,
                            interval=validated_interval.value,
                            run_id=run_id,
                            requested_start_time=start_time,
                            requested_end_time=end_time,
                            effective_start_time=resume_start_time,
                            effective_end_time=effective_end_time,
                            pages_fetched=0,
                            records_fetched=0,
                            records_stored=0,
                            gap_report=gap_report,
                            execution_time_seconds=round(time.perf_counter() - start_perf, 4),
                            status="already_up_to_date",
                        )
                    logger.info(
                        "Tail Resume: Resuming %s %s from %d.",
                        sym,
                        validated_interval.value,
                        resume_start_time,
                    )
                    current_start_time = resume_start_time

            effective_start_time = current_start_time

            # Pagination Loop
            while current_start_time <= effective_end_time:
                klines = self._fetch_klines_page(
                    symbol=sym,
                    interval=validated_interval,
                    start_time=current_start_time,
                    end_time=effective_end_time,
                    limit=1000,
                )
                pages_fetched += 1
                records_fetched += len(klines)

                if not klines:
                    # Range exhausted
                    break

                # Forming-candle exclusion: filter out any candle with close_time_ms >= server_time_ms
                closed_klines = [
                    k for k in klines
                    if getattr(k, "close_time_ms", getattr(k, "close_time", 0)) < server_time_ms
                ]

                # Edge case: If every candle in the batch is forming, terminate safely
                if not closed_klines:
                    logger.info(
                        "All %d returned candles are forming candles. Terminating pagination safely.",
                        len(klines),
                    )
                    break

                # Transaction Boundary 1: Store canonical raw reconstruction
                canonical_payload = json.dumps([k.raw for k in klines], separators=(",", ":"))
                fetched_at_ms = int(time.time() * 1000)
                params_dict = {
                    "symbol": sym,
                    "interval": validated_interval.value,
                    "startTime": current_start_time,
                    "endTime": effective_end_time,
                    "limit": 1000,
                }
                params_json = json.dumps(params_dict, separators=(",", ":"))

                raw_response_id = self._db.insert_raw_api_response(
                    endpoint="/api/v3/klines",
                    response_json=canonical_payload,
                    fetched_at=fetched_at_ms,
                    params_json=params_json,
                    ingestion_run_id=run_id,
                )

                # Prepare normalized rows referencing raw_response_id
                kline_rows: List[KlineRow] = []
                for k in closed_klines:
                    ot = getattr(k, "open_time_ms", getattr(k, "open_time", 0))
                    ct = getattr(k, "close_time_ms", getattr(k, "close_time", 0))
                    trades_cnt = getattr(k, "trades", getattr(k, "number_of_trades", 0))
                    tbv = getattr(k, "taker_buy_base_asset_volume", getattr(k, "taker_buy_base_volume", 0))
                    tqv = getattr(k, "taker_buy_quote_asset_volume", getattr(k, "taker_buy_quote_volume", 0))
                    kline_rows.append({
                        "symbol": sym,
                        "interval": validated_interval.value,
                        "open_time": ot,
                        "open_price": k.open,
                        "high_price": k.high,
                        "low_price": k.low,
                        "close_price": k.close,
                        "volume": k.volume,
                        "close_time": ct,
                        "quote_asset_volume": k.quote_asset_volume,
                        "number_of_trades": trades_cnt,
                        "taker_buy_base_volume": tbv,
                        "taker_buy_quote_volume": tqv,
                        "raw_response_id": raw_response_id,
                    })

                # Transaction Boundary 2: Normalized batch upsert
                stored_batch_count = self._db.batch_upsert_klines(kline_rows)
                records_stored += stored_batch_count
                all_stored_klines.extend(closed_klines)

                # Advance next start time
                last_closed_kline = closed_klines[-1]
                last_ct = getattr(last_closed_kline, "close_time_ms", getattr(last_closed_kline, "close_time", 0))
                last_ot = getattr(last_closed_kline, "open_time_ms", getattr(last_closed_kline, "open_time", 0))
                next_start_time = last_ct + 1

                # Loop Advancement Assertion (prevent pagination stalls)
                if next_start_time <= current_start_time:
                    raise RuntimeError(
                        f"Pagination stalled: next_start_time ({next_start_time}) "
                        f"did not advance past current_start_time ({current_start_time})."
                    )

                current_start_time = next_start_time

                # Check termination bounds
                if (
                    len(klines) < 1000
                    or len(closed_klines) < len(klines)
                    or last_ot >= effective_end_time
                    or current_start_time > effective_end_time
                ):
                    break

                # Conservative pacing & adaptive telemetry check
                self._handle_pacing_and_telemetry(self._request_delay_ms)

            # Continuity analysis across the examined sequence
            gap_report = detect_kline_gaps(
                symbol=sym,
                interval=validated_interval,
                klines=all_stored_klines,
                expected_start_time=effective_start_time if all_stored_klines else None,
                expected_end_time=effective_end_time if all_stored_klines else None,
            )

            # Mark ingestion run completed
            self._db.update_ingestion_run(
                run_id=run_id,
                status="completed",
                completed_at=int(time.time() * 1000),
                records_fetched=records_fetched,
                records_stored=records_stored,
            )

            return DownloadResult(
                symbol=sym,
                interval=validated_interval.value,
                run_id=run_id,
                requested_start_time=start_time,
                requested_end_time=end_time,
                effective_start_time=effective_start_time,
                effective_end_time=effective_end_time,
                pages_fetched=pages_fetched,
                records_fetched=records_fetched,
                records_stored=records_stored,
                gap_report=gap_report,
                execution_time_seconds=round(time.perf_counter() - start_perf, 4),
                status="completed",
            )

        except Exception as exc:
            self._db.update_ingestion_run(
                run_id=run_id,
                status="failed",
                completed_at=int(time.time() * 1000),
                records_fetched=records_fetched,
                records_stored=records_stored,
                error_message=str(exc),
            )
            raise
