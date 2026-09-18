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
from trade_intelligence.db.exceptions import DatabaseError, DatabaseIntegrityError

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
    "DatabaseIntegrityError",
]
