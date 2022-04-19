from dataclasses import dataclass
from datetime import datetime

import torch

from loss import SimpleLossCompute, Batch_Loss, SimpleLossCompute_tst, Test_Loss
from pgportfolio.marketdata.datamatricesnew import DataMatricesNew

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"


@dataclass
class Parameters:
    total_step = 100000
    x_window_size = 48
    batch_size = 128
    feature_number = 4
    output_step = 2
    model_index = 2
    multihead_num = 2
    local_context_length = 5
    model_dim = 12

    test_portion = 0.08
    validation_portion = 0.08
    trading_consumption = 0.0025
    variance_penalty = 0.0
    cost_penalty = 0.0
    learning_rate = 0.001
    weight_decay = 1e-7
    daily_interest_rate = 0.001

    model_name = 'RAT'

    @property
    def interest_rate(self):
        return self.daily_interest_rate  / 24 / 2


FLAGS = Parameters()

DM = DataMatricesNew(dataset_file='./data/stocks/12_stocks_30min_2022-0419.nc',
                     feature_number=FLAGS.feature_number,
                     window_size=FLAGS.x_window_size,
                     is_permed=False,
                     buffer_bias_ratio=5e-5,
                     batch_size=FLAGS.batch_size,  # 128,
                     validation_portion=FLAGS.validation_portion,
                     test_portion=FLAGS.test_portion,
                     portion_reversed=False)