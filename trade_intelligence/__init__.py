"""Trade Intelligence - Binance Market Data Engine."""

from trade_intelligence.binance.client import BinanceRestClient
from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.binance.exceptions import (
    BinanceApiError,
    BinanceClientError,
    BinanceConnectionError,
    BinanceHttpError,
    BinanceResponseError,
    BinanceTimeoutError,
)
from trade_intelligence.db.database import Database
from trade_intelligence.db.exceptions import (
    DatabaseError,
    DatabaseInitError,
    DatabaseIntegrityError,
)
from trade_intelligence.klines.downloader import HistoricalKlineDownloader
from trade_intelligence.klines.types import DownloadResult
from trade_intelligence.universe.sync import SyncResult, UniverseSyncer

__version__ = "0.1.0"

__all__ = [
    "BinanceRestClient",
    "KlineInterval",
    "BinanceClientError",
    "BinanceConnectionError",
    "BinanceTimeoutError",
    "BinanceHttpError",
    "BinanceApiError",
    "BinanceResponseError",
    "Database",
    "DatabaseError",
    "DatabaseInitError",
    "DatabaseIntegrityError",
    "SyncResult",
    "UniverseSyncer",
    "HistoricalKlineDownloader",
    "DownloadResult",
]
