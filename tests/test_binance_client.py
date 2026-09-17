"""Unit tests for BinanceRestClient using mocked HTTP responses."""

from decimal import Decimal
from typing import Any, Optional
import unittest
from unittest.mock import MagicMock, patch
import requests

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


def _make_mock_response(
    status_code: int = 200, json_data: Optional[Any] = None, text: str = ""
) -> MagicMock:
    """Helper to construct a mock requests.Response."""
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = status_code
    mock_resp.ok = 200 <= status_code < 300
    mock_resp.text = text if text else ("" if json_data is None else str(json_data))
    if json_data is not None:
        mock_resp.json.return_value = json_data
    else:
        mock_resp.json.side_effect = ValueError("No JSON")
    return mock_resp


SAMPLE_RAW_KLINE = [
    1499040000000,      # 0: Open time
    "0.01634790",       # 1: Open
    "0.80000000",       # 2: High
    "0.01575800",       # 3: Low
    "0.01577100",       # 4: Close
    "148976.11427815",  # 5: Volume
    1499644799999,      # 6: Close time
    "2434.19055334",    # 7: Quote asset volume
    308,                # 8: Number of trades
    "1756.87402397",    # 9: Taker buy base asset volume
    "28.46694368",      # 10: Taker buy quote asset volume
    "0",                # 11: Ignore
]


class TestBinanceRestClient(unittest.TestCase):
    """Unit test suite for Binance REST client."""

    def setUp(self):
        self.client = BinanceRestClient(base_url="https://api.binance.test", timeout=5.0)

    def tearDown(self):
        self.client.close()

    def test_configurable_base_url_and_context_manager(self):
        """Test custom base URL stripping and context manager."""
        with BinanceRestClient(base_url="https://custom.api.binance.com/") as client:
            self.assertEqual(client.base_url, "https://custom.api.binance.com")
            self.assertEqual(client.timeout, 10.0)

    @patch.object(requests.Session, "request")
    def test_get_server_time_success(self, mock_request):
        """Verify server time retrieval and normalization."""
        mock_request.return_value = _make_mock_response(
            json_data={"serverTime": 1710000000123}
        )

        result = self.client.get_server_time()

        self.assertIsInstance(result, ServerTime)
        self.assertEqual(result.server_time_ms, 1710000000123)
        self.assertEqual(result.raw["serverTime"], 1710000000123)
        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.binance.test/api/v3/time",
            params=None,
            timeout=5.0,
        )

    @patch.object(requests.Session, "request")
    def test_get_exchange_info_success(self, mock_request):
        """Verify exchange info retrieval and normalization."""
        payload = {
            "timezone": "UTC",
            "serverTime": 1710000000000,
            "rateLimits": [{"rateLimitType": "REQUEST_WEIGHT", "limit": 1200}],
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "baseAssetPrecision": 8,
                    "quoteAsset": "USDT",
                    "quotePrecision": 8,
                    "quoteAssetPrecision": 8,
                    "isSpotTradingAllowed": True,
                    "isMarginTradingAllowed": True,
                }
            ],
        }
        mock_request.return_value = _make_mock_response(json_data=payload)

        result = self.client.get_exchange_info(symbol="BTCUSDT")

        self.assertIsInstance(result, ExchangeInfo)
        self.assertEqual(result.timezone, "UTC")
        self.assertEqual(result.server_time_ms, 1710000000000)
        self.assertIn("BTCUSDT", result.symbols)

        sym = result.symbols["BTCUSDT"]
        self.assertIsInstance(sym, SymbolInfo)
        self.assertEqual(sym.symbol, "BTCUSDT")
        self.assertEqual(sym.base_asset, "BTC")
        self.assertEqual(sym.quote_asset, "USDT")
        self.assertTrue(sym.is_spot_trading_allowed)
        self.assertEqual(sym.raw["symbol"], "BTCUSDT")

        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.binance.test/api/v3/exchangeInfo",
            params={"symbol": "BTCUSDT"},
            timeout=5.0,
        )

    @patch.object(requests.Session, "request")
    def test_get_exchange_info_multiple_symbols(self, mock_request):
        """Verify exchange info with multiple symbol filter."""
        payload = {"timezone": "UTC", "serverTime": 1710000000000, "symbols": []}
        mock_request.return_value = _make_mock_response(json_data=payload)

        self.client.get_exchange_info(symbols=["BTCUSDT", "ETHUSDT"])
        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.binance.test/api/v3/exchangeInfo",
            params={"symbols": '["BTCUSDT","ETHUSDT"]'},
            timeout=5.0,
        )

    @patch.object(requests.Session, "request")
    def test_get_ticker_price_single_symbol(self, mock_request):
        """Verify single symbol ticker price retrieval."""
        mock_request.return_value = _make_mock_response(
            json_data={"symbol": "BTCUSDT", "price": "67890.50"}
        )

        result = self.client.get_ticker_price("BTCUSDT")

        self.assertIsInstance(result, PriceTicker)
        self.assertEqual(result.symbol, "BTCUSDT")
        self.assertIsInstance(result.price, Decimal)
        self.assertEqual(result.price, Decimal("67890.50"))
        self.assertEqual(result.raw["price"], "67890.50")
        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.binance.test/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
            timeout=5.0,
        )

    @patch.object(requests.Session, "request")
    def test_get_ticker_price_all_symbols(self, mock_request):
        """Verify all tickers retrieval when symbol is None."""
        mock_request.return_value = _make_mock_response(
            json_data=[
                {"symbol": "BTCUSDT", "price": "67890.50"},
                {"symbol": "ETHUSDT", "price": "3500.25"},
            ]
        )

        result = self.client.get_ticker_price()

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].symbol, "BTCUSDT")
        self.assertIsInstance(result[0].price, Decimal)
        self.assertEqual(result[0].price, Decimal("67890.50"))
        self.assertEqual(result[1].symbol, "ETHUSDT")
        self.assertIsInstance(result[1].price, Decimal)
        self.assertEqual(result[1].price, Decimal("3500.25"))

    @patch.object(requests.Session, "request")
    def test_get_klines_all_five_supported_intervals(self, mock_request):
        """Verify klines retrieval across all 5 supported intervals."""
        intervals = ["5m", "15m", "1h", "4h", "1d"]

        for interval in intervals:
            with self.subTest(interval=interval):
                mock_request.return_value = _make_mock_response(
                    json_data=[SAMPLE_RAW_KLINE]
                )

                klines = self.client.get_klines(symbol="BTCUSDT", interval=interval)

                self.assertIsInstance(klines, list)
                self.assertEqual(len(klines), 1)
                kline = klines[0]
                self.assertIsInstance(kline, Kline)
                self.assertEqual(kline.open_time_ms, 1499040000000)
                self.assertIsInstance(kline.open, Decimal)
                self.assertEqual(kline.open, Decimal("0.01634790"))
                self.assertIsInstance(kline.high, Decimal)
                self.assertEqual(kline.high, Decimal("0.80000000"))
                self.assertIsInstance(kline.low, Decimal)
                self.assertEqual(kline.low, Decimal("0.01575800"))
                self.assertIsInstance(kline.close, Decimal)
                self.assertEqual(kline.close, Decimal("0.01577100"))
                self.assertIsInstance(kline.volume, Decimal)
                self.assertEqual(kline.volume, Decimal("148976.11427815"))
                self.assertEqual(kline.close_time_ms, 1499644799999)
                self.assertIsInstance(kline.quote_asset_volume, Decimal)
                self.assertEqual(kline.quote_asset_volume, Decimal("2434.19055334"))
                self.assertEqual(kline.trades, 308)
                self.assertIsInstance(kline.taker_buy_base_asset_volume, Decimal)
                self.assertEqual(kline.taker_buy_base_asset_volume, Decimal("1756.87402397"))
                self.assertIsInstance(kline.taker_buy_quote_asset_volume, Decimal)
                self.assertEqual(kline.taker_buy_quote_asset_volume, Decimal("28.46694368"))
                self.assertEqual(kline.raw, SAMPLE_RAW_KLINE)

    @patch.object(requests.Session, "request")
    def test_get_klines_with_enum_interval(self, mock_request):
        """Verify klines retrieval using KlineInterval enum."""
        mock_request.return_value = _make_mock_response(json_data=[SAMPLE_RAW_KLINE])

        klines = self.client.get_klines(
            symbol="BTCUSDT", interval=KlineInterval.INTERVAL_4H
        )
        self.assertEqual(len(klines), 1)
        mock_request.assert_called_with(
            method="GET",
            url="https://api.binance.test/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "4h"},
            timeout=5.0,
        )

    @patch.object(requests.Session, "request")
    def test_get_klines_time_range_and_limit(self, mock_request):
        """Verify klines supports historical parameters (startTime, endTime, limit)."""
        mock_request.return_value = _make_mock_response(json_data=[SAMPLE_RAW_KLINE])

        self.client.get_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time=1600000000000,
            end_time=1600003600000,
            limit=50,
        )

        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.binance.test/api/v3/klines",
            params={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "startTime": 1600000000000,
                "endTime": 1600003600000,
                "limit": 50,
            },
            timeout=5.0,
        )

    def test_get_klines_unsupported_interval(self):
        """Verify unsupported interval raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.client.get_klines(symbol="BTCUSDT", interval="3m")
        self.assertIn("Unsupported kline interval", str(ctx.exception))

        with self.assertRaises(ValueError):
            self.client.get_klines(symbol="BTCUSDT", interval="1w")

    @patch.object(requests.Session, "request")
    def test_timeout_handling(self, mock_request):
        """Verify request timeout is caught and raises BinanceTimeoutError."""
        mock_request.side_effect = requests.exceptions.Timeout("Read timed out")

        with self.assertRaises(BinanceTimeoutError):
            self.client.get_server_time()

    @patch.object(requests.Session, "request")
    def test_connection_error_handling(self, mock_request):
        """Verify connection error raises BinanceConnectionError."""
        mock_request.side_effect = requests.exceptions.ConnectionError("DNS failure")

        with self.assertRaises(BinanceConnectionError):
            self.client.get_server_time()

    @patch.object(requests.Session, "request")
    def test_binance_api_error_handling(self, mock_request):
        """Verify structured Binance API errors are parsed into BinanceApiError."""
        mock_request.return_value = _make_mock_response(
            status_code=400,
            json_data={"code": -1121, "msg": "Invalid symbol."},
        )

        with self.assertRaises(BinanceApiError) as ctx:
            self.client.get_ticker_price("INVALID")

        err = ctx.exception
        self.assertEqual(err.code, -1121)
        self.assertEqual(err.msg, "Invalid symbol.")
        self.assertEqual(err.status_code, 400)
        self.assertIn("-1121", str(err))

    @patch.object(requests.Session, "request")
    def test_http_error_without_json_handling(self, mock_request):
        """Verify non-JSON HTTP errors raise BinanceHttpError."""
        mock_request.return_value = _make_mock_response(
            status_code=502,
            text="<html>502 Bad Gateway</html>",
        )

        with self.assertRaises(BinanceHttpError) as ctx:
            self.client.get_server_time()

        self.assertEqual(ctx.exception.status_code, 502)

    @patch.object(requests.Session, "request")
    def test_malformed_json_response_handling(self, mock_request):
        """Verify malformed JSON on 200 raises BinanceResponseError."""
        mock_request.return_value = _make_mock_response(
            status_code=200,
            text="invalid json string",
        )

        with self.assertRaises(BinanceResponseError):
            self.client.get_server_time()

    @patch.object(requests.Session, "request")
    def test_malformed_model_data_handling(self, mock_request):
        """Verify unexpected schema structure raises BinanceResponseError."""
        # Missing serverTime in dict
        mock_request.return_value = _make_mock_response(
            json_data={"wrongField": 123}
        )
        with self.assertRaises(BinanceResponseError):
            self.client.get_server_time()

        # Kline with insufficient elements
        mock_request.return_value = _make_mock_response(
            json_data=[[1499040000000, "0.01"]]  # fewer than 11 elements
        )
        with self.assertRaises(BinanceResponseError):
            self.client.get_klines("BTCUSDT", "1h")

    def test_decimal_precision_preservation(self):
        """Verify high-precision decimals and binary-float problematic values (0.1, satoshis)."""
        # 0.1 problem in IEEE-754: 0.1 + 0.2 != 0.3 in float
        ticker1 = PriceTicker.from_raw({"symbol": "TEST1", "price": "0.1"})
        ticker2 = PriceTicker.from_raw({"symbol": "TEST2", "price": "0.2"})
        self.assertIsInstance(ticker1.price, Decimal)
        self.assertIsInstance(ticker2.price, Decimal)
        self.assertEqual(ticker1.price + ticker2.price, Decimal("0.3"))
        self.assertEqual(str(ticker1.price), "0.1")

        # 8-decimal satoshi / low-cap crypto price
        ticker_sats = PriceTicker.from_raw({"symbol": "SHIBUSDT", "price": "0.00000001"})
        self.assertEqual(ticker_sats.price, Decimal("0.00000001"))
        self.assertEqual(f"{ticker_sats.price:f}", "0.00000001")
        self.assertEqual(ticker_sats.raw["price"], "0.00000001")

        # High precision kline values
        raw_kline = list(SAMPLE_RAW_KLINE)
        raw_kline[1] = "0.00000007"
        raw_kline[4] = "0.00000009"
        kline = Kline.from_raw(raw_kline)
        self.assertEqual(kline.open, Decimal("0.00000007"))
        self.assertEqual(kline.close, Decimal("0.00000009"))
        self.assertEqual(kline.close - kline.open, Decimal("0.00000002"))

    @patch.object(requests.Session, "request")
    def test_get_klines_limit_validation(self, mock_request):
        """Verify klines limit parameter range validation (1 <= limit <= 1000)."""
        # Invalid limits should raise ValueError before making HTTP call
        with self.assertRaises(ValueError) as ctx:
            self.client.get_klines("BTCUSDT", "1h", limit=0)
        self.assertIn("Invalid limit: 0", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self.client.get_klines("BTCUSDT", "1h", limit=1001)
        self.assertIn("Invalid limit: 1001", str(ctx.exception))

        with self.assertRaises(ValueError):
            self.client.get_klines("BTCUSDT", "1h", limit=-5)

        mock_request.assert_not_called()

        # Valid boundaries: limit=1 and limit=1000
        mock_request.return_value = _make_mock_response(json_data=[SAMPLE_RAW_KLINE])

        self.client.get_klines("BTCUSDT", "1h", limit=1)
        mock_request.assert_called_with(
            method="GET",
            url="https://api.binance.test/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1h", "limit": 1},
            timeout=5.0,
        )

        self.client.get_klines("BTCUSDT", "1h", limit=1000)
        mock_request.assert_called_with(
            method="GET",
            url="https://api.binance.test/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1h", "limit": 1000},
            timeout=5.0,
        )

        # limit=None preserves default behavior (no limit param sent)
        self.client.get_klines("BTCUSDT", "1h", limit=None)
        mock_request.assert_called_with(
            method="GET",
            url="https://api.binance.test/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1h"},
            timeout=5.0,
        )

    @patch.object(requests.Session, "request")
    def test_non_integer_binance_api_error_handling(self, mock_request):
        """Verify unexpected non-integer API error code does not leak ValueError."""
        mock_request.return_value = _make_mock_response(
            status_code=504,
            json_data={"code": "GATEWAY_TIMEOUT", "msg": "Upstream timeout"},
        )

        with self.assertRaises(BinanceApiError) as ctx:
            self.client.get_server_time()

        err = ctx.exception
        self.assertEqual(err.code, "GATEWAY_TIMEOUT")
        self.assertEqual(err.msg, "Upstream timeout")
        self.assertEqual(err.status_code, 504)
        self.assertIn("[GATEWAY_TIMEOUT] Upstream timeout", str(err))


if __name__ == "__main__":
    unittest.main()
