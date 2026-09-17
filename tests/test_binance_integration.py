"""Live integration and connectivity tests against Binance public REST API.

These tests make real network requests to Binance's public REST API.
No API keys or authentication required.
"""

import unittest
from trade_intelligence.binance.client import BinanceRestClient
from trade_intelligence.binance.enums import KlineInterval
from trade_intelligence.binance.exceptions import BinanceApiError, BinanceTimeoutError
from trade_intelligence.binance.models import ExchangeInfo, Kline, PriceTicker, ServerTime


class TestBinanceIntegration(unittest.TestCase):
    """Live connectivity tests for Binance public REST API."""

    @classmethod
    def setUpClass(cls):
        cls.client = BinanceRestClient(timeout=15.0)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_live_server_time(self):
        """Verify real Binance server time retrieval."""
        result = self.client.get_server_time()
        self.assertIsInstance(result, ServerTime)
        self.assertIsInstance(result.server_time_ms, int)
        # Server time should be well past year 2023 (1.7e12 ms)
        self.assertGreater(result.server_time_ms, 1700000000000)
        self.assertIn("serverTime", result.raw)

    def test_live_exchange_info_btcusdt(self):
        """Verify real Binance Spot exchange information for BTCUSDT."""
        info = self.client.get_exchange_info(symbol="BTCUSDT")
        self.assertIsInstance(info, ExchangeInfo)
        self.assertIn("BTCUSDT", info.symbols)

        btc = info.symbols["BTCUSDT"]
        self.assertEqual(btc.symbol, "BTCUSDT")
        self.assertEqual(btc.base_asset, "BTC")
        self.assertEqual(btc.quote_asset, "USDT")
        self.assertEqual(btc.status, "TRADING")
        self.assertIsInstance(btc.is_spot_trading_allowed, bool)

    def test_live_ticker_price_btcusdt(self):
        """Verify real Binance ticker price for BTCUSDT."""
        ticker = self.client.get_ticker_price("BTCUSDT")
        self.assertIsInstance(ticker, PriceTicker)
        self.assertEqual(ticker.symbol, "BTCUSDT")
        self.assertIsInstance(ticker.price, float)
        self.assertGreater(ticker.price, 0.0)
        self.assertIn("price", ticker.raw)

    def test_live_klines_all_five_intervals_btcusdt(self):
        """Verify real Binance klines retrieval for BTCUSDT across all 5 supported intervals."""
        intervals = ["5m", "15m", "1h", "4h", "1d"]

        for interval in intervals:
            with self.subTest(interval=interval):
                klines = self.client.get_klines(
                    symbol="BTCUSDT",
                    interval=interval,
                    limit=5,
                )
                self.assertIsInstance(klines, list)
                self.assertEqual(len(klines), 5)

                for kline in klines:
                    self.assertIsInstance(kline, Kline)
                    self.assertGreater(kline.open_time_ms, 0)
                    self.assertGreater(kline.close_time_ms, kline.open_time_ms)
                    self.assertGreater(kline.open, 0.0)
                    self.assertGreater(kline.high, 0.0)
                    self.assertGreater(kline.low, 0.0)
                    self.assertGreater(kline.close, 0.0)
                    self.assertGreaterEqual(kline.high, kline.low)
                    self.assertGreaterEqual(kline.volume, 0.0)
                    self.assertIsInstance(kline.raw, list)
                    self.assertGreaterEqual(len(kline.raw), 11)

    def test_live_invalid_symbol_raises_binance_api_error(self):
        """Verify that querying an invalid symbol on live API raises BinanceApiError."""
        with self.assertRaises(BinanceApiError) as ctx:
            self.client.get_ticker_price("INVALID_SYMBOL_XYZ123")

        self.assertEqual(ctx.exception.code, -1121)
        self.assertIn("Invalid symbol", ctx.exception.msg)

    def test_live_timeout_handling(self):
        """Verify timeout exception handling when timeout threshold is near zero."""
        client = BinanceRestClient(timeout=0.000001)
        try:
            with self.assertRaises(BinanceTimeoutError):
                client.get_server_time()
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
