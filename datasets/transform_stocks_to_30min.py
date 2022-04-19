import os
from datetime import timedelta

import pandas as pd

src_datadir = "../data/stocks/12_stocks_1min_2022-0419"
dst_datadir = "../data/stocks/12_stocks_30min_2022-0419"

src_frequency = timedelta(minutes=1)
dst_frequency = timedelta(minutes=30)

os.makedirs(dst_datadir, exist_ok=True)

for filename in os.listdir(src_datadir):
    symbol = filename[:filename.find(".")]
    symbol_df = pd.read_csv(f"{src_datadir}/{filename}", parse_dates=['time'])

    # Convert to start time
    symbol_df['time'] = symbol_df['time'] - src_frequency

    resampled = symbol_df.set_index('time').resample('30T').aggregate(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).reset_index()
    resampled = resampled.dropna()

    # Convert back to end time
    resampled['time'] = resampled['time'] + dst_frequency

    resampled.to_csv(f"{dst_datadir}/{filename}", index=False)
