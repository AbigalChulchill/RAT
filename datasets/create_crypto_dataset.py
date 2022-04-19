import os

import pandas as pd
import xarray as xr

datadir = "../data/crypto/cash_USDT_5min_2020-04-19_2022-04-19_old"

features = ['high', 'low', 'open', 'close', 'volume', 'weighted_average']
coins = []
times = set()
for filename in os.listdir(datadir):
    coin = filename[:filename.find(".")]
    coins.append(coin)
    coin_df = pd.read_csv(f"{datadir}/{filename}")

    times.update(coin_df['time'].tolist())

time_index = sorted(times)

da = xr.DataArray(coords=[features, coins, time_index], dims=['features', 'coins', 'time'])

for filename in os.listdir(datadir):
    coin = filename[:filename.find(".")]
    coin_df = pd.read_csv(f"{datadir}/{filename}").rename({'weightedAverage': 'weighted_average'}, axis=1)

    for feature in features:
        da.loc[feature, coin, coin_df['time'].tolist()] = coin_df[feature]


pass