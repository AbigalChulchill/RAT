import logging
import os
from datetime import datetime, timezone, time
import pandas as pd

from pgportfolio.marketdata.coinlist import CoinList

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

start_date_str = '2020-04-19'
end_date_str = '2022-04-19'
period = 300  # 5 minutes
cash_coin = 'USDT'
number_of_coins = 12

start_time_unix = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()
end_time_unix = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()

volume_average_days = 1
volume_forward = 0
coin_list = CoinList(cash_coin, end_time_unix, volume_average_days, volume_forward)

# coins_volume = coin_list.topNVolume(number_of_coins)
# coins = list(coins_volume.index)

coins = ['BTC', 'ETC', 'BNB', 'XRP', 'QNT', 'DASH', 'KSM', 'HT', 'WPR', 'HDAO', 'KEY', 'MET']

period_in_min = int(period / 60)

# datadir = f"../data/crypto/cash_{cash_coin}_{period_in_min}min_{start_date_str}_{end_date_str}"
datadir = f"../data/crypto/cash_{cash_coin}_{period_in_min}min_{start_date_str}_{end_date_str}"
os.makedirs(datadir, exist_ok=True)

for coin in coins:
    batch_start = start_time_unix

    logging.info(f"Dumping data for pair {cash_coin}_{coin}")

    coin_charts = []
    while batch_start < end_time_unix:
        batch_end = batch_start + 30 * 24 * 3600
        if batch_end > end_time_unix:
            batch_end = end_time_unix

        batch_start_str = datetime.fromtimestamp(batch_start).strftime('%Y-%m-%d %H:%M')
        batch_end_str = datetime.fromtimestamp(batch_end).strftime('%Y-%m-%d %H:%M')

        logging.info(f"Dumping data from {batch_start_str} till {batch_end_str}")

        chart = coin_list.get_chart_until_success(
            pair=coin_list.allActiveCoins.at[coin, 'pair'],
            start=batch_start,
            end=batch_end - 1,  # The end time in Poloniex is inclusive
            period=period)
        coin_charts.extend(chart)

        batch_start = batch_end

        time.sleep(5)

    df_chart = pd.DataFrame(coin_charts)
    df_chart['date'] = pd.to_datetime(df_chart['date'], unit='s')
    df_chart.rename({'date': 'time'}, axis=1, inplace=True)
    df_chart.sort_values('time', inplace=True)

    df_chart.to_csv(f"{datadir}/{coin}.csv", index=False)

pass
