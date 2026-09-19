"""Historical kline downloader and continuity analysis package."""

from trade_intelligence.klines.downloader import HistoricalKlineDownloader
from trade_intelligence.klines.gap_detector import (
    detect_kline_gaps,
    interval_to_milliseconds,
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
]
