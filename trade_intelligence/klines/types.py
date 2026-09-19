"""Type definitions and result structures for the historical kline downloader."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class KlineGap:
    """Represents a detected chronological gap in a sequence of klines."""

    symbol: str
    interval: str
    gap_start_ms: int
    gap_end_ms: int
    missing_candles: int


@dataclass(frozen=True)
class GapReport:
    """Detailed summary of data continuity across an examined sequence of klines.

    Note:
        GapReport reports continuity strictly within the candle sequence examined
        by the current download operation. When tail resuming, it does not scan
        prior database history.
    """

    symbol: str
    interval: str
    total_expected_candles: int
    total_actual_candles: int
    total_missing_candles: int
    coverage_ratio: float
    gaps: List[KlineGap] = field(default_factory=list)


@dataclass(frozen=True)
class DownloadResult:
    """Diagnostic and audit results returned by HistoricalKlineDownloader.

    Attributes:
        symbol: Trading pair symbol.
        interval: Kline interval (e.g. '1h').
        run_id: Ingestion run ID in the database.
        requested_start_time: Initial start time requested by caller (epoch ms).
        requested_end_time: Initial end time requested by caller (epoch ms).
        effective_start_time: Actual start time used (after alignment/tail resume).
        effective_end_time: Actual clamped end time used (capped before server time).
        pages_fetched: Total API pages (requests) retrieved.
        records_fetched: Total raw klines returned by API.
        records_stored: Total klines successfully persisted into the database.
        gap_report: Continuity analysis of the examined sequence.
        execution_time_seconds: Total wall-clock execution duration in seconds.
        status: Operation status ('completed', 'already_up_to_date', 'failed').
            'completed' indicates the requested API download work finished successfully.
            It does NOT imply the range is proven gap-free.
    """

    symbol: str
    interval: str
    run_id: int
    requested_start_time: int
    requested_end_time: int
    effective_start_time: int
    effective_end_time: int
    pages_fetched: int
    records_fetched: int
    records_stored: int
    gap_report: GapReport
    execution_time_seconds: float
    status: str
