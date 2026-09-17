"""Binance Spot public REST API client."""

import json
from typing import Any, Dict, List, Optional, Union
import requests

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
)

DEFAULT_BASE_URL = "https://api.binance.com"
DEFAULT_TIMEOUT = 10.0


class BinanceRestClient:
    """Client for Binance Spot public REST API endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ):
        """Initialize the Binance Spot REST client.

        Args:
            base_url: Base URL for Binance REST API. Defaults to 'https://api.binance.com'.
            timeout: Request timeout in seconds. Defaults to 10.0.
            session: Optional pre-configured requests.Session instance.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._custom_session = session is not None
        self._session = session or requests.Session()

    def __enter__(self) -> "BinanceRestClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP session if managed internally."""
        if not self._custom_session and self._session:
            self._session.close()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute an HTTP request against the Binance REST API with error handling.

        Args:
            method: HTTP method (e.g. 'GET').
            endpoint: API path (e.g. '/api/v3/time').
            params: Query parameters dictionary.

        Returns:
            Parsed JSON response from Binance.

        Raises:
            BinanceConnectionError: If network connection fails.
            BinanceTimeoutError: If request times out.
            BinanceApiError: If Binance returns an application error response.
            BinanceHttpError: If HTTP status indicates an error without structured API payload.
            BinanceResponseError: If response is not valid JSON.
        """
        url = f"{self.base_url}{endpoint}"

        # Clean params removing None values
        clean_params = (
            {k: v for k, v in params.items() if v is not None} if params else None
        )

        try:
            response = self._session.request(
                method=method,
                url=url,
                params=clean_params,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise BinanceTimeoutError(
                f"Request to {endpoint} timed out after {self.timeout}s"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise BinanceConnectionError(
                f"Connection to {url} failed: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise BinanceClientError(
                f"Unexpected request failure for {url}: {exc}"
            ) from exc

        # Attempt to parse JSON body
        try:
            data = response.json()
        except Exception:
            # Response was not JSON
            if not response.ok:
                raise BinanceHttpError(
                    status_code=response.status_code,
                    message=response.text or "Unknown HTTP error",
                    raw_response=response.text,
                )
            raise BinanceResponseError(
                f"Failed to parse JSON response from {endpoint}: {response.text[:200]}"
            )

        # Check for Binance API errors (typically with HTTP >= 400 or containing "code" and "msg")
        if not response.ok or (isinstance(data, dict) and "code" in data and "msg" in data and data["code"] != 0):
            if isinstance(data, dict) and "code" in data and "msg" in data:
                raw_code = data["code"]
                try:
                    code: Union[int, str] = int(raw_code)
                except (ValueError, TypeError):
                    code = str(raw_code)
                raise BinanceApiError(
                    code=code,
                    msg=str(data["msg"]),
                    status_code=response.status_code,
                    raw_response=data,
                )
            raise BinanceHttpError(
                status_code=response.status_code,
                message=str(data),
                raw_response=str(data),
            )

        return data

    def get_server_time(self) -> ServerTime:
        """Retrieve Binance server time.

        Returns:
            Normalized ServerTime instance containing server_time_ms and raw dict.
        """
        data = self._request("GET", "/api/v3/time")
        return ServerTime.from_raw(data)

    def get_exchange_info(
        self,
        symbol: Optional[str] = None,
        symbols: Optional[List[str]] = None,
    ) -> ExchangeInfo:
        """Retrieve Spot exchange information and trading rules.

        Args:
            symbol: Optional single symbol filter (e.g. 'BTCUSDT').
            symbols: Optional list of symbol filters (e.g. ['BTCUSDT', 'ETHUSDT']).

        Returns:
            Normalized ExchangeInfo instance.
        """
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        elif symbols:
            params["symbols"] = json.dumps([s.upper() for s in symbols], separators=(",", ":"))

        data = self._request("GET", "/api/v3/exchangeInfo", params=params)
        return ExchangeInfo.from_raw(data)

    def get_ticker_price(
        self,
        symbol: Optional[str] = None,
    ) -> Union[PriceTicker, List[PriceTicker]]:
        """Retrieve current latest price for a symbol or all symbols.

        Args:
            symbol: Optional symbol filter (e.g. 'BTCUSDT'). If omitted, returns all tickers.

        Returns:
            PriceTicker if symbol was provided, or List[PriceTicker] if omitted.
        """
        params = {"symbol": symbol.upper()} if symbol else None
        data = self._request("GET", "/api/v3/ticker/price", params=params)

        if isinstance(data, list):
            return [PriceTicker.from_raw(item) for item in data]
        return PriceTicker.from_raw(data)

    def get_klines(
        self,
        symbol: str,
        interval: Union[str, KlineInterval],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Kline]:
        """Retrieve historical or current Kline/OHLCV candlestick data.

        Args:
            symbol: Trading pair symbol (e.g. 'BTCUSDT').
            interval: Kline interval ('5m', '15m', '1h', '4h', '1d' or KlineInterval enum).
            start_time: Optional start timestamp in milliseconds.
            end_time: Optional end timestamp in milliseconds.
            limit: Optional number of klines to retrieve (default 500, max 1000).

        Returns:
            List of normalized Kline instances.

        Raises:
            ValueError: If the interval is not in the supported set.
            BinanceClientError: For network, HTTP, or API errors.
        """
        validated_interval = KlineInterval.from_value(interval)

        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": validated_interval.value,
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        if limit is not None:
            if not isinstance(limit, int) or limit < 1 or limit > 1000:
                raise ValueError(
                    f"Invalid limit: {limit}. Limit must be an integer between 1 and 1000."
                )
            params["limit"] = int(limit)

        data = self._request("GET", "/api/v3/klines", params=params)

        if not isinstance(data, list):
            raise BinanceResponseError(f"Expected list of klines, got {type(data).__name__}")

        return [Kline.from_raw(item) for item in data]
