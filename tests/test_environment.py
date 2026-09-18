"""Sanity checks for environment and dependencies."""

import unittest
import sys
import requests
import trade_intelligence


class TestEnvironment(unittest.TestCase):
    """Basic environment sanity checks."""

    def test_python_version(self):
        """Ensure Python version is 3.10 or greater."""
        self.assertGreaterEqual(sys.version_info[:2], (3, 10))

    def test_requests_importable(self):
        """Ensure requests library is installed and importable."""
        self.assertTrue(hasattr(requests, "__version__"))

    def test_package_importable(self):
        """Ensure trade_intelligence package is importable."""
        self.assertEqual(trade_intelligence.__version__, "0.1.0")

    def test_database_exceptions_exported(self):
        """Ensure all database exception classes are exported at package root."""
        self.assertTrue(hasattr(trade_intelligence, "DatabaseError"))
        self.assertTrue(hasattr(trade_intelligence, "DatabaseInitError"))
        self.assertTrue(hasattr(trade_intelligence, "DatabaseIntegrityError"))


if __name__ == "__main__":
    unittest.main()
