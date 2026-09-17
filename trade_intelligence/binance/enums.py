"""Enumerations and constants for Binance data access."""

from enum import Enum
from typing import Union


class KlineInterval(str, Enum):
    """Supported Kline/Candlestick intervals."""

    INTERVAL_5M = "5m"
    INTERVAL_15M = "15m"
    INTERVAL_1H = "1h"
    INTERVAL_4H = "4h"
    INTERVAL_1D = "1d"

    @classmethod
    def from_value(cls, val: Union[str, "KlineInterval"]) -> "KlineInterval":
        """Validate and convert a string or enum to a valid KlineInterval."""
        if isinstance(val, cls):
            return val
        if isinstance(val, str):
            for member in cls:
                if member.value == val:
                    return member
        supported = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unsupported kline interval: '{val}'. Supported intervals are: {supported}"
        )
