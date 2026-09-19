"""Type definitions, enums, and data models for multi-timeframe orchestration and gap repair."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.klines.types import DownloadResult


class DownloadStatus(str, Enum):
    """Execution status for historical kline download."""

    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    FAILED = "failed"


class CoverageStatus(str, Enum):
    """Data completeness status for a requested historical range."""

    NOT_CHECKED = "not_checked"
    COMPLETE = "complete"
    GAPS_FOUND = "gaps_found"


class RepairStatus(str, Enum):
    """Gap repair status for an examined range."""

    NOT_REQUESTED = "not_requested"
    NOT_NEEDED = "not_needed"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class GapType(str, Enum):
    """Categorization of a detected continuity gap."""

    LEADING = "leading"
    INTERIOR = "interior"
    TRAILING = "trailing"
    FULL_RANGE = "full_range"


class PipelineStatus(str, Enum):
    """Overall execution status of the multi-timeframe pipeline."""

    COMPLETED = "completed"
    COMPLETED_WITH_GAPS = "completed_with_gaps"
    FAILED = "failed"
    PARTIAL_FAILURE = "partial_failure"


@dataclass(frozen=True)
class CoverageGap:
    """A specific missing interval in the requested range.

    Both ``gap_start_open_time_ms`` and ``gap_end_open_time_ms`` are
    **inclusive** open_time values of missing candles.  Neither field
    ever represents a close_time.
    """

    symbol: str
    interval: str
    gap_type: GapType
    gap_start_open_time_ms: int
    gap_end_open_time_ms: int
    missing_candles: int


@dataclass(frozen=True)
class CoverageReport:
    """Detailed summary of candle coverage across a full requested range."""

    symbol: str
    interval: str
    requested_start_ms: int
    requested_end_ms: int
    aligned_start_ms: int
    aligned_end_ms: int
    expected_candles: int
    actual_candles: int
    missing_candles: int
    coverage_ratio: float
    is_complete: bool
    leading_gaps: List[CoverageGap] = field(default_factory=list)
    interior_gaps: List[CoverageGap] = field(default_factory=list)
    trailing_gaps: List[CoverageGap] = field(default_factory=list)
    full_range_gaps: List[CoverageGap] = field(default_factory=list)
    all_gaps: List[CoverageGap] = field(default_factory=list)


@dataclass(frozen=True)
class RepairSegment:
    """Targeted download segment calculated to repair a detected gap.

    Both ``segment_start_open_time_ms`` and ``segment_end_open_time_ms``
    are **inclusive** open_time values of missing candles.  Translation
    to the downloader's request semantics happens at the repair boundary.
    """

    symbol: str
    interval: str
    segment_start_open_time_ms: int
    segment_end_open_time_ms: int
    expected_missing_candles: int
    source_gap: CoverageGap


@dataclass(frozen=True)
class RepairSegmentResult:
    """Result of an individual segment repair download attempt."""

    segment: RepairSegment
    download_result: Optional[DownloadResult]
    candles_recovered: int
    success: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class TimeframeResult:
    """Complete diagnostic and operational result for a single (symbol, interval)."""

    symbol: str
    interval: str
    requested_start_time: int
    requested_end_time: int
    download_status: DownloadStatus
    coverage_status: CoverageStatus
    repair_status: RepairStatus
    stored_candles: int
    expected_candles: int
    initial_coverage: Optional[CoverageReport] = None
    final_coverage: Optional[CoverageReport] = None
    repairs: List[RepairSegmentResult] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass(frozen=True)
class PipelineConfig:
    """Orchestration-level settings.

    Does **not** contain ``request_delay_ms`` or ``max_retries`` — those
    belong to the Phase 1.4.2 ``HistoricalKlineDownloader`` constructor.
    """

    fail_fast: bool = False
    repair_interior_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.fail_fast, bool):
            raise TypeError(f"fail_fast must be bool, got {type(self.fail_fast).__name__}")
        if not isinstance(self.repair_interior_only, bool):
            raise TypeError(
                f"repair_interior_only must be bool, got {type(self.repair_interior_only).__name__}"
            )


def validate_pipeline_request(request: "PipelineRequest") -> None:
    """Validate all fields of a PipelineRequest strictly.

    Raises TypeError or ValueError if any field is invalid.
    """
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

    if not isinstance(request.download_before_scan, bool):
        raise TypeError(
            f"download_before_scan must be bool, got {type(request.download_before_scan).__name__}"
        )
    if not isinstance(request.repair_gaps, bool):
        raise TypeError(
            f"repair_gaps must be bool, got {type(request.repair_gaps).__name__}"
        )

    if request.config is not None and not isinstance(request.config, PipelineConfig):
        raise TypeError(
            f"config must be PipelineConfig or None, got {type(request.config).__name__}"
        )


@dataclass(frozen=True)
class PipelineRequest:
    """Structured request to orchestrate historical ingestion and coverage."""

    symbols: Sequence[str]
    intervals: Sequence[Union[str, KlineInterval]]
    start_time: int
    end_time: int
    download_before_scan: bool = True
    repair_gaps: bool = True
    config: Optional[PipelineConfig] = None

    def __post_init__(self) -> None:
        validate_pipeline_request(self)


@dataclass(frozen=True)
class PipelineResult:
    """Aggregate result of a multi-timeframe pipeline run."""

    overall_status: PipelineStatus
    timeframe_results: Dict[Tuple[str, str], TimeframeResult]
    total_symbols: int
    total_intervals: int
    completed_count: int
    failed_count: int
    total_repairs_attempted: int
    total_repairs_completed: int
    execution_time_seconds: float
    errors: List[str] = field(default_factory=list)
