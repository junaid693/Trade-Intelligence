"""Binance API access module."""

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
from trade_intelligence.binance.models import (
    ExchangeInfo,
    Kline,
    PriceTicker,
    ServerTime,
    SymbolInfo,
)

__all__ = [
    "BinanceRestClient",
    "KlineInterval",
    "BinanceClientError",
    "BinanceConnectionError",
    "BinanceTimeoutError",
    "BinanceHttpError",
    "BinanceApiError",
    "BinanceResponseError",
    "ServerTime",
    "SymbolInfo",
    "ExchangeInfo",
    "PriceTicker",
    "Kline",
]
