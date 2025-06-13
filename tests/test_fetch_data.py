import unittest
import os
import pandas as pd
from unittest.mock import patch, MagicMock

from data.fetch_data import hent_binance_data

class TestFetchData(unittest.TestCase):
    @patch("data.fetch_data.ccxt.binance")
    def test_hent_binance_data_creates_csv(self, mock_binance_class):
        # Mock binance.fetch_ohlcv til at returnere dummy OHLCV data
        mock_binance = MagicMock()
        mock_binance.fetch_ohlcv.return_value = [
            [1609459200000, 29000, 29300, 28800, 29200, 1.2],  # (eksempel-data)
            [1609462800000, 29200, 29500, 29100, 29400, 2.1],
        ]
        mock_binance_class.return_value = mock_binance

        # Fjern evt. gammel fil
        testfile = "test_BTCUSDT_1h.csv"
        if os.path.exists(testfile):
            os.remove(testfile)

        # Kald funktionen
        hent_binance_data(symbol="BTC/USDT", timeframe="1h", limit=2, filnavn=testfile)
        # Tjek at filen nu findes
        self.assertTrue(os.path.exists(testfile))
        # Indlæs og check indhold (nu med sep=";")
        df = pd.read_csv(testfile, sep=";")
        self.assertEqual(len(df), 2)
        self.assertIn("close", df.columns)

        # Oprydning
        os.remove(testfile)

if __name__ == "__main__":
    unittest.main()
