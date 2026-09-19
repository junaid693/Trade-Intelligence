"""Historical kline downloader, continuity analysis, and orchestration pipeline."""

from trade_intelligence.klines.downloader import HistoricalKlineDownloader
from trade_intelligence.klines.gap_detector import (
    detect_kline_gaps,
    interval_to_milliseconds,
)
from trade_intelligence.klines.pipeline import (
    CoverageReport,
    CoverageScanner,
    GapRepairer,
    HistoricalDataOrchestrator,
    PipelineRequest,
    PipelineResult,
    TimeframeResult,
)
from trade_intelligence.klines.types import (
    DownloadResult,
    GapReport,
    KlineGap,
)

__all__ = [
    "HistoricalKlineDownloader",
    "DownloadResult",
    "GapReport",
    "KlineGap",
    "detect_kline_gaps",
    "interval_to_milliseconds",
    "CoverageReport",
    "CoverageScanner",
    "GapRepairer",
    "HistoricalDataOrchestrator",
    "PipelineRequest",
    "PipelineResult",
    "TimeframeResult",
]
