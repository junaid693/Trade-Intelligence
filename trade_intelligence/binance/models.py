"""Typed Python data models for normalized Binance REST API responses."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional
from trade_intelligence.binance.exceptions import BinanceResponseError


@dataclass(frozen=True)
class ServerTime:
    """Normalized Binance server time response."""

    server_time_ms: int
    raw: Dict[str, Any]

    @classmethod
    def from_raw(cls, data: Any) -> "ServerTime":
        if not isinstance(data, dict) or "serverTime" not in data:
            raise BinanceResponseError(f"Expected dict with 'serverTime', got: {type(data).__name__}")
        try:
            return cls(server_time_ms=int(data["serverTime"]), raw=data)
        except (ValueError, TypeError) as exc:
            raise BinanceResponseError(f"Invalid serverTime value: {data.get('serverTime')}") from exc


@dataclass(frozen=True)
class SymbolInfo:
    """Normalized metadata for a single Binance Spot trading pair."""

    symbol: str
    status: str
    base_asset: str
    base_asset_precision: int
    quote_asset: str
    quote_precision: int
    quote_asset_precision: int
    is_spot_trading_allowed: bool
    is_margin_trading_allowed: bool
    raw: Dict[str, Any]

    @classmethod
    def from_raw(cls, data: Any) -> "SymbolInfo":
        if not isinstance(data, dict):
            raise BinanceResponseError(f"Expected symbol info dict, got: {type(data).__name__}")
        try:
            return cls(
                symbol=str(data["symbol"]),
                status=str(data.get("status", "")),
                base_asset=str(data.get("baseAsset", "")),
                base_asset_precision=int(data.get("baseAssetPrecision", 8)),
                quote_asset=str(data.get("quoteAsset", "")),
                quote_precision=int(data.get("quotePrecision", 8)),
                quote_asset_precision=int(data.get("quoteAssetPrecision", 8)),
                is_spot_trading_allowed=bool(data.get("isSpotTradingAllowed", False)),
                is_margin_trading_allowed=bool(data.get("isMarginTradingAllowed", False)),
                raw=data,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise BinanceResponseError(f"Failed to parse symbol info: {exc}") from exc


@dataclass(frozen=True)
class ExchangeInfo:
    """Normalized exchange metadata and trading rules."""

    timezone: str
    server_time_ms: int
    symbols: Dict[str, SymbolInfo]
    rate_limits: List[Dict[str, Any]]
    raw: Dict[str, Any]

    @classmethod
    def from_raw(cls, data: Any) -> "ExchangeInfo":
        if not isinstance(data, dict):
            raise BinanceResponseError(f"Expected exchangeInfo dict, got: {type(data).__name__}")
        try:
            raw_symbols = data.get("symbols", [])
            symbols_map = {
                s_dict["symbol"]: SymbolInfo.from_raw(s_dict)
                for s_dict in raw_symbols
                if isinstance(s_dict, dict) and "symbol" in s_dict
            }
            return cls(
                timezone=str(data.get("timezone", "UTC")),
                server_time_ms=int(data.get("serverTime", 0)),
                symbols=symbols_map,
                rate_limits=list(data.get("rateLimits", [])),
                raw=data,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise BinanceResponseError(f"Failed to parse exchange info: {exc}") from exc


@dataclass(frozen=True)
class PriceTicker:
    """Normalized price ticker."""

    symbol: str
    price: Decimal
    raw: Dict[str, Any]

    @classmethod
    def from_raw(cls, data: Any) -> "PriceTicker":
        if not isinstance(data, dict) or "symbol" not in data or "price" not in data:
            raise BinanceResponseError(f"Expected dict with 'symbol' and 'price', got: {data}")
        try:
            return cls(
                symbol=str(data["symbol"]),
                price=Decimal(str(data["price"])),
                raw=data,
            )
        except (ValueError, TypeError, InvalidOperation) as exc:
            raise BinanceResponseError(f"Failed to parse price ticker: {exc}") from exc


@dataclass(frozen=True)
class Kline:
    """Normalized Kline / OHLCV candlestick data."""

    open_time_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time_ms: int
    quote_asset_volume: Decimal
    trades: int
    taker_buy_base_asset_volume: Decimal
    taker_buy_quote_asset_volume: Decimal
    raw: List[Any]

    @classmethod
    def from_raw(cls, data: Any) -> "Kline":
        if not isinstance(data, (list, tuple)) or len(data) < 11:
            raise BinanceResponseError(
                f"Expected list/tuple with at least 11 elements for kline, got {data}"
            )
        try:
            return cls(
                open_time_ms=int(data[0]),
                open=Decimal(str(data[1])),
                high=Decimal(str(data[2])),
                low=Decimal(str(data[3])),
                close=Decimal(str(data[4])),
                volume=Decimal(str(data[5])),
                close_time_ms=int(data[6]),
                quote_asset_volume=Decimal(str(data[7])),
                trades=int(data[8]),
                taker_buy_base_asset_volume=Decimal(str(data[9])),
                taker_buy_quote_asset_volume=Decimal(str(data[10])),
                raw=list(data),
            )
        except (ValueError, TypeError, IndexError, InvalidOperation) as exc:
            raise BinanceResponseError(f"Failed to parse kline entry: {exc}") from exc
