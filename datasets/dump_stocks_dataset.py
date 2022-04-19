import logging
import os

import pandas as pd
import time
from datetime import datetime

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# API_KEY = '0Y03HCRH4IQYJGHT'
API_KEY = "SOHC081J9R3FBHE2"
BASE_URL = 'https://www.alphavantage.co/query'
FUNCTION_NAME = 'TIME_SERIES_INTRADAY_EXTENDED'

symbols = [
    'SPY',  # S&P 500
    'AAL',  # American Airlines
    'AAPL',  # Apple
    'AMD',  # AMD
    'AA',  # Alcoa
    'INTC',  # Intel Corp
    'JPM',  # JP Morgan
    'PINS',  # Pinterest
    'RIO',  # Rio Tinto
    'TEVA',  # Teva
    'TRVG',  # Trivago
    'TSLA',  # Tesla
]

period = 1  # in minutes
requests_per_minute = 5

current_time_str = datetime.now().strftime("%Y-%m%-d")

datadir = f"../data/stocks/12_stocks_{period}min_{current_time_str}"

os.makedirs(datadir, exist_ok=True)

for symbol in symbols:
    csv_file = f"{datadir}/{symbol}.csv"
    if os.path.exists(csv_file):
        logging.info(f"Data for stock {symbol} has already been dumped, so will skip it in this run")
        continue

    logging.info(f"Dumping data for {symbol}")

    symbol_dfs = []
    for year in range(1, 3):
        for month in range(1, 13):
            history_slice = f"year{year}month{month}"
            logging.info(f"Dumping slice {history_slice}")

            slice_df = pd.read_csv(
                f'{BASE_URL}?function={FUNCTION_NAME}&symbol={symbol}&interval={period}min&slice={history_slice}&apikey={API_KEY}')

            symbol_dfs.append(slice_df)
            time.sleep((60.0 / requests_per_minute) + 1)

    df = pd.concat(symbol_dfs)
    df.sort_values('time', inplace=True)
    df.to_csv(csv_file, index=False)
