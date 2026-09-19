"""Gap repair planning and targeted segment download execution."""

import logging
from typing import List, Optional, Tuple, Union

from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.klines.downloader import HistoricalKlineDownloader
from trade_intelligence.klines.gap_detector import interval_to_milliseconds
from trade_intelligence.klines.pipeline.coverage import CoverageScanner
from trade_intelligence.klines.pipeline.types import (
    CoverageGap,
    CoverageReport,
    GapType,
    RepairSegment,
    RepairSegmentResult,
    RepairStatus,
)

logger = logging.getLogger(__name__)


class GapRepairer:
    """Plans and executes targeted downloads to fill detected continuity gaps.

    Design constraints (Phase 1.4.3):
    - Zero retry logic.  All retries are delegated to ``HistoricalKlineDownloader``.
    - The repairer never calls Binance directly.
    - Segment boundaries are inclusive open_time values; translation to the
      downloader's request semantics happens at the download call boundary.
    """

    @staticmethod
    def plan_repair_segments(
        report: CoverageReport,
        interior_only: bool = False,
    ) -> List[RepairSegment]:
        """Translate detected gaps into ordered repair segments.

        Args:
            report: A ``CoverageReport`` from ``CoverageScanner``.
            interior_only: If ``True`` only ``INTERIOR`` gaps produce
                segments (``LEADING``, ``TRAILING``, and ``FULL_RANGE``
                gaps are excluded).

        Returns:
            Ordered list of ``RepairSegment`` values — one per gap.
        """
        if not isinstance(report, CoverageReport):
            raise TypeError(
                f"report must be a CoverageReport, got {type(report).__name__}"
            )

        segments: List[RepairSegment] = []
        for gap in report.all_gaps:
            if interior_only and gap.gap_type != GapType.INTERIOR:
                continue
            segments.append(
                RepairSegment(
                    symbol=gap.symbol,
                    interval=gap.interval,
                    segment_start_open_time_ms=gap.gap_start_open_time_ms,
                    segment_end_open_time_ms=gap.gap_end_open_time_ms,
                    expected_missing_candles=gap.missing_candles,
                    source_gap=gap,
                )
            )
        return segments

    @staticmethod
    def execute_repairs(
        segments: List[RepairSegment],
        downloader: HistoricalKlineDownloader,
    ) -> List[RepairSegmentResult]:
        """Execute targeted downloads for each repair segment.

        For each segment the method translates inclusive open_time
        boundaries into the downloader's ``start_time`` / ``end_time``
        parameters and delegates the full download + persist lifecycle
        to ``HistoricalKlineDownloader.download_historical_klines``.

        Zero retry logic — if the downloader raises after exhausting its
        own retry policy the result is recorded as ``success=False`` and
        execution continues to the next segment.

        Args:
            segments: Ordered repair segments from :meth:`plan_repair_segments`.
            downloader: Pre-constructed ``HistoricalKlineDownloader`` instance.

        Returns:
            Ordered list of ``RepairSegmentResult`` — one per input segment.
        """
        if not isinstance(downloader, HistoricalKlineDownloader):
            raise TypeError(
                f"downloader must be a HistoricalKlineDownloader, got {type(downloader).__name__}"
            )

        results: List[RepairSegmentResult] = []
        for seg in segments:
            interval_ms = interval_to_milliseconds(seg.interval)

            # Translate inclusive open_time boundaries into the downloader's
            # request semantics.  start_time is the open_time of the first
            # missing candle.  end_time must cover the close of the last
            # missing candle so that the downloader's ≤ filter includes it.
            dl_start = seg.segment_start_open_time_ms
            dl_end = seg.segment_end_open_time_ms + interval_ms - 1

            logger.info(
                "Repairing gap %s/%s: open_time [%d .. %d] -> download range [%d .. %d] (%d expected candles)",
                seg.symbol,
                seg.interval,
                seg.segment_start_open_time_ms,
                seg.segment_end_open_time_ms,
                dl_start,
                dl_end,
                seg.expected_missing_candles,
            )

            try:
                dr = downloader.download_historical_klines(
                    symbol=seg.symbol,
                    interval=seg.interval,
                    start_time=dl_start,
                    end_time=dl_end,
                    resume=False,
                )
                results.append(
                    RepairSegmentResult(
                        segment=seg,
                        download_result=dr,
                        candles_recovered=dr.records_stored,
                        success=True,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Repair segment failed for %s/%s [%d..%d]: %s",
                    seg.symbol,
                    seg.interval,
                    seg.segment_start_open_time_ms,
                    seg.segment_end_open_time_ms,
                    exc,
                )
                results.append(
                    RepairSegmentResult(
                        segment=seg,
                        download_result=None,
                        candles_recovered=0,
                        success=False,
                        error=str(exc),
                    )
                )
        return results

    @staticmethod
    def verify_repairs(
        symbol: str,
        interval: Union[str, KlineInterval],
        start_time: int,
        end_time: int,
        scanner: CoverageScanner,
        server_time_ms: Optional[int] = None,
    ) -> Tuple[CoverageReport, RepairStatus]:
        """Re-scan the specific requested range and determine repair status.

        Args:
            symbol: Trading pair symbol.
            interval: Kline interval.
            start_time: Requested inclusive start (epoch ms).
            end_time: Requested inclusive end (epoch ms).
            scanner: ``CoverageScanner`` instance.
            server_time_ms: Optional current server time for clamping.

        Returns:
            Tuple of (updated ``CoverageReport``, ``RepairStatus``).
        """
        report = scanner.scan_coverage(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            server_time_ms=server_time_ms,
        )

        if report.is_complete:
            return report, RepairStatus.COMPLETED
        return report, RepairStatus.INCOMPLETE
