"""Multi-timeframe historical data orchestrator.

Determines what candles need downloading, verifies coverage, plans and
executes gap repairs, and verifies post-repair continuity — all using
the existing Phase 1.4.2 ``HistoricalKlineDownloader`` as the sole
owner of Binance network interactions.
"""

import logging
import time
from typing import Dict, List, Sequence, Tuple, Union

from trade_intelligence.binance.client import BinanceRestClient
from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.db.database import Database
from trade_intelligence.klines.downloader import HistoricalKlineDownloader
from trade_intelligence.klines.pipeline.coverage import CoverageScanner
from trade_intelligence.klines.pipeline.repair import GapRepairer
from trade_intelligence.klines.pipeline.types import (
    CoverageStatus,
    DownloadStatus,
    PipelineConfig,
    PipelineRequest,
    PipelineResult,
    PipelineStatus,
    RepairStatus,
    TimeframeResult,
)

logger = logging.getLogger(__name__)


class HistoricalDataOrchestrator:
    """Orchestrates multi-timeframe historical ingestion and gap repair.

    Execution order is strictly sequential: symbol-first, then interval.
    ``BTCUSDT (1d → 4h → 1h → 15m → 5m) → ETHUSDT (1d → 4h → …)``

    The orchestrator receives a **pre-constructed**
    ``HistoricalKlineDownloader`` instance and never configures pacing,
    retries, or rate-limit parameters.
    """

    def __init__(
        self,
        db: Database,
        client: BinanceRestClient,
        downloader: HistoricalKlineDownloader,
    ) -> None:
        if not isinstance(db, Database):
            raise TypeError(f"db must be a Database instance, got {type(db).__name__}")
        if not isinstance(client, BinanceRestClient):
            raise TypeError(f"client must be a BinanceRestClient, got {type(client).__name__}")
        if not isinstance(downloader, HistoricalKlineDownloader):
            raise TypeError(
                f"downloader must be a HistoricalKlineDownloader, got {type(downloader).__name__}"
            )

        self._db = db
        self._client = client
        self._downloader = downloader
        self._scanner = CoverageScanner(db)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_request(request: PipelineRequest) -> None:
        """Raise ``TypeError`` / ``ValueError`` on invalid request fields."""
        if not isinstance(request, PipelineRequest):
            raise TypeError(f"request must be a PipelineRequest, got {type(request).__name__}")

        if not request.symbols or not isinstance(request.symbols, (list, tuple)):
            raise ValueError("symbols must be a non-empty list or tuple")
        for s in request.symbols:
            if not isinstance(s, str) or not s.strip():
                raise ValueError(f"Each symbol must be a non-empty string, got: {s!r}")

        if not request.intervals or not isinstance(request.intervals, (list, tuple)):
            raise ValueError("intervals must be a non-empty list or tuple")
        for iv in request.intervals:
            KlineInterval.from_value(iv)  # raises ValueError on invalid

        if isinstance(request.start_time, bool) or not isinstance(request.start_time, int):
            raise TypeError(f"start_time must be int, got {type(request.start_time).__name__}")
        if request.start_time < 0:
            raise ValueError(f"start_time must be non-negative, got {request.start_time}")

        if isinstance(request.end_time, bool) or not isinstance(request.end_time, int):
            raise TypeError(f"end_time must be int, got {type(request.end_time).__name__}")
        if request.end_time < 0:
            raise ValueError(f"end_time must be non-negative, got {request.end_time}")

        if request.start_time >= request.end_time:
            raise ValueError(
                f"start_time ({request.start_time}) must be < end_time ({request.end_time})"
            )

    # ------------------------------------------------------------------
    # Single-pair execution
    # ------------------------------------------------------------------

    def _run_single_timeframe(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: int,
        end_time: int,
        request: PipelineRequest,
        config: PipelineConfig,
    ) -> TimeframeResult:
        """Execute download → scan → repair → verify for one (symbol, interval)."""
        iv_str = interval.value
        download_status = DownloadStatus.NOT_STARTED
        coverage_status = CoverageStatus.NOT_CHECKED
        repair_status = RepairStatus.NOT_REQUESTED

        initial_coverage = None
        final_coverage = None
        repairs: list = []
        error_message = None
        stored = 0
        expected = 0

        try:
            # --- 1. optional initial download ---
            if request.download_before_scan:
                logger.info("Downloading %s/%s [%d..%d]", symbol, iv_str, start_time, end_time)
                try:
                    dl_result = self._downloader.download_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=start_time,
                        end_time=end_time,
                        resume=True,
                    )
                    download_status = DownloadStatus.COMPLETED
                    stored = dl_result.records_stored
                except Exception as dl_exc:
                    download_status = DownloadStatus.FAILED
                    error_message = f"Download failed: {dl_exc}"
                    logger.warning("Download failed for %s/%s: %s", symbol, iv_str, dl_exc)

            # --- 2. coverage scan ---
            server_time_ms: int | None = None
            try:
                server_time_ms = self._client.get_server_time().server_time_ms
            except Exception:
                pass  # non-fatal; proceed without forming-candle clamping

            initial_coverage = self._scanner.scan_coverage(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                server_time_ms=server_time_ms,
            )
            expected = initial_coverage.expected_candles
            stored = initial_coverage.actual_candles

            if initial_coverage.is_complete:
                coverage_status = CoverageStatus.COMPLETE
                repair_status = RepairStatus.NOT_NEEDED
            else:
                coverage_status = CoverageStatus.GAPS_FOUND

                # --- 3. optional gap repair ---
                if request.repair_gaps:
                    segments = GapRepairer.plan_repair_segments(
                        report=initial_coverage,
                        interior_only=config.repair_interior_only,
                    )

                    if not segments:
                        repair_status = RepairStatus.NOT_NEEDED
                    else:
                        logger.info(
                            "Repairing %d segments for %s/%s",
                            len(segments),
                            symbol,
                            iv_str,
                        )
                        repairs = GapRepairer.execute_repairs(
                            segments=segments,
                            downloader=self._downloader,
                        )

                        # --- 4. post-repair verification (specific range) ---
                        final_coverage, repair_status = GapRepairer.verify_repairs(
                            symbol=symbol,
                            interval=interval,
                            start_time=start_time,
                            end_time=end_time,
                            scanner=self._scanner,
                            server_time_ms=server_time_ms,
                        )
                        stored = final_coverage.actual_candles
                        expected = final_coverage.expected_candles

                        if final_coverage.is_complete:
                            coverage_status = CoverageStatus.COMPLETE
                else:
                    repair_status = RepairStatus.NOT_REQUESTED

        except Exception as exc:
            error_message = str(exc)
            logger.error("Error processing %s/%s: %s", symbol, iv_str, exc)
            if download_status == DownloadStatus.NOT_STARTED:
                download_status = DownloadStatus.FAILED
            if repair_status not in (RepairStatus.NOT_REQUESTED, RepairStatus.NOT_NEEDED):
                repair_status = RepairStatus.FAILED

        return TimeframeResult(
            symbol=symbol,
            interval=iv_str,
            requested_start_time=start_time,
            requested_end_time=end_time,
            download_status=download_status,
            coverage_status=coverage_status,
            repair_status=repair_status,
            stored_candles=stored,
            expected_candles=expected,
            initial_coverage=initial_coverage,
            final_coverage=final_coverage,
            repairs=repairs,
            error_message=error_message,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, request: PipelineRequest) -> PipelineResult:
        """Execute the full multi-timeframe orchestration pipeline.

        Execution is sequential: symbol-first, then intervals in the
        order supplied in the request.

        Args:
            request: Structured ``PipelineRequest``.

        Returns:
            ``PipelineResult`` with per-pair diagnostics and aggregate stats.

        Raises:
            TypeError / ValueError: On invalid ``PipelineRequest`` fields.
        """
        self._validate_request(request)

        config = request.config or PipelineConfig()
        start_wall = time.monotonic()

        # Normalise inputs
        symbols: List[str] = [s.strip().upper() for s in request.symbols]
        intervals: List[KlineInterval] = [
            KlineInterval.from_value(iv) for iv in request.intervals
        ]

        results: Dict[Tuple[str, str], TimeframeResult] = {}
        errors: List[str] = []
        completed = 0
        failed = 0
        total_repairs_attempted = 0
        total_repairs_completed = 0

        for sym in symbols:
            for iv in intervals:
                logger.info("Pipeline: processing %s / %s", sym, iv.value)

                tf_result = self._run_single_timeframe(
                    symbol=sym,
                    interval=iv,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    request=request,
                    config=config,
                )

                results[(sym, iv.value)] = tf_result

                if tf_result.error_message:
                    errors.append(f"{sym}/{iv.value}: {tf_result.error_message}")

                if tf_result.download_status == DownloadStatus.FAILED:
                    failed += 1
                else:
                    completed += 1

                total_repairs_attempted += len(tf_result.repairs)
                total_repairs_completed += sum(
                    1 for r in tf_result.repairs if r.success
                )

                # fail_fast
                if config.fail_fast and tf_result.error_message:
                    logger.warning(
                        "fail_fast enabled — aborting pipeline after %s/%s failure",
                        sym,
                        iv.value,
                    )
                    break
            else:
                continue
            break  # break outer loop if inner break triggered

        elapsed = time.monotonic() - start_wall

        # Determine overall status
        if failed == 0 and all(
            r.coverage_status == CoverageStatus.COMPLETE for r in results.values()
        ):
            overall = PipelineStatus.COMPLETED
        elif failed == 0:
            overall = PipelineStatus.COMPLETED_WITH_GAPS
        elif completed > 0:
            overall = PipelineStatus.PARTIAL_FAILURE
        else:
            overall = PipelineStatus.FAILED

        return PipelineResult(
            overall_status=overall,
            timeframe_results=results,
            total_symbols=len(symbols),
            total_intervals=len(intervals),
            completed_count=completed,
            failed_count=failed,
            total_repairs_attempted=total_repairs_attempted,
            total_repairs_completed=total_repairs_completed,
            execution_time_seconds=round(elapsed, 3),
            errors=errors,
        )
