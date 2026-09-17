"""Custom exception hierarchy for the Binance REST client."""

from typing import Any, Optional, Union


class BinanceClientError(Exception):
    """Base exception for all Binance client-related errors."""


class BinanceConnectionError(BinanceClientError):
    """Raised when network connectivity fails (DNS, socket, connection refused)."""


class BinanceTimeoutError(BinanceClientError):
    """Raised when an API request times out."""


class BinanceHttpError(BinanceClientError):
    """Raised when Binance returns an HTTP error status code (4xx, 5xx)."""

    def __init__(self, status_code: int, message: str, raw_response: Optional[str] = None):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.raw_response = raw_response


class BinanceApiError(BinanceHttpError):
    """Raised when Binance returns a structured error payload (code and msg)."""

    def __init__(
        self,
        code: Union[int, str],
        msg: str,
        status_code: int = 400,
        raw_response: Optional[Any] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=f"[{code}] {msg}",
            raw_response=str(raw_response) if raw_response is not None else None,
        )
        self.code = code
        self.msg = msg
        self.api_data = raw_response


class BinanceResponseError(BinanceClientError):
    """Raised when the response payload is malformed or cannot be normalized."""
