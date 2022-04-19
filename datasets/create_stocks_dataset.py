import os
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr

datadir = "../data/stocks/12_stocks_30min_2022-0419"
output_file = f"{datadir}.nc"

features = ['high', 'low', 'open', 'close', 'volume']
symbols = []
times = set()

frequency_minutes = 30

for filename in os.listdir(datadir):
    symbol = filename[:filename.find(".")]
    symbols.append(symbol)
    symbol_df = pd.read_csv(f"{datadir}/{filename}")

    times.update(symbol_df['time'].tolist())

dates = sorted(set(map(lambda d: datetime.fromisoformat(d).date(), times)))

time_index = []
for date in dates:
    time = 0
    for i in range(16 * int(60.0 / frequency_minutes)):  # minutes
        time_index.append(datetime.combine(date, (datetime.min + timedelta(hours=4, minutes=(i + 1) * frequency_minutes)).time()))

da = xr.DataArray(coords=[features, symbols, time_index], dims=['features', 'coins', 'time'])

for filename in os.listdir(datadir):
    symbol = filename[:filename.find(".")]
    symbol_df = pd.read_csv(f"{datadir}/{filename}", parse_dates=['time']) \
        .rename({'weightedAverage': 'weighted_average'}, axis=1)

    for feature in features:
        valid_times = symbol_df['time'].isin(time_index)

        da.loc[feature, symbol, symbol_df[valid_times]['time'].tolist()] = symbol_df[valid_times][feature]

da.to_netcdf(output_file)
