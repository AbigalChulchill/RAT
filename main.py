import os

import torch

from loss import SimpleLossCompute, SimpleLossCompute_tst, Batch_Loss, Test_Loss
from pgportfolio.marketdata.datamatrices import DataMatrices
from pgportfolio.tools.configprocess import parse_time
from rat.rat import make_model
# import matplotlib.pyplot as plt
# import seaborn
# seaborn.set_context(context="talk")
from utils.train_test_utils import train_net, test_net

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--total_step', type=int, default=80000)
parser.add_argument('--x_window_size', type=int, default=31)
# parser.add_argument('--y_window_size', type=int, default=11)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--coin_num', type=int, default=11)
parser.add_argument('--feature_number', type=int, default=4)
parser.add_argument('--output_step', type=int, default=500)
parser.add_argument('--model_index', type=int, default=0)
parser.add_argument('--multihead_num', type=int, default=2)
parser.add_argument('--local_context_length', type=int, default=5)
parser.add_argument('--model_dim', type=int, default=12)

parser.add_argument('--test_portion', type=float, default=0.08)
parser.add_argument('--trading_consumption', type=float, default=0.0025)
parser.add_argument('--variance_penalty', type=float, default=0.0)
parser.add_argument('--cost_penalty', type=float, default=0.0)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=5e-8)
parser.add_argument('--daily_interest_rate', type=float, default=0.001)

parser.add_argument('--start', type=str, default="2016/01/01")
parser.add_argument('--end', type=str, default="2018/01/01")
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--model_dir', type=str, default=None)

FLAGS = parser.parse_args()


class NoamOpt:
    "Optim wrapper that implements rate."

    # 512, 1, 400
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if self.warmup == 0:
            return self.factor
        else:
            return self.factor * \
                   (self.model_size ** (-0.5) *
                    min(step ** (-0.5), step * self.warmup ** (-1.5)))



start = parse_time(FLAGS.start)
end = parse_time(FLAGS.end)
DM = DataMatrices(start=start, end=end,
                  market="poloniex",
                  feature_number=FLAGS.feature_number,
                  window_size=FLAGS.x_window_size,
                  online=True,
                  period=1800,
                  coin_filter=11,
                  is_permed=False,
                  buffer_bias_ratio=5e-5,
                  batch_size=FLAGS.batch_size,  # 128,
                  volume_average_days=30,
                  test_portion=FLAGS.test_portion,  # 0.08,
                  portion_reversed=False)

#################set learning rate###################
lr_model_sz = 5120
factor = FLAGS.learning_rate  # 1.0
warmup = 0  # 800

total_step = FLAGS.total_step
x_window_size = FLAGS.x_window_size  # 31

batch_size = FLAGS.batch_size
coin_num = FLAGS.coin_num  # 11
feature_number = FLAGS.feature_number  # 4
trading_consumption = FLAGS.trading_consumption  # 0.0025
variance_penalty = FLAGS.variance_penalty  # 0 #0.01
cost_penalty = FLAGS.cost_penalty  # 0 #0.01
output_step = FLAGS.output_step  # 50
local_context_length = FLAGS.local_context_length
model_dim = FLAGS.model_dim
weight_decay = FLAGS.weight_decay
interest_rate = FLAGS.daily_interest_rate / 24 / 2

model = make_model(batch_size, coin_num, x_window_size, feature_number,
                   N=1, d_model_Encoder=FLAGS.multihead_num * model_dim,
                   d_model_Decoder=FLAGS.multihead_num * model_dim,
                   d_ff_Encoder=FLAGS.multihead_num * model_dim,
                   d_ff_Decoder=FLAGS.multihead_num * model_dim,
                   h=FLAGS.multihead_num,
                   dropout=0.01,
                   local_context_length=local_context_length)

# model = make_model3(N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
# TODO: Uncomment for GPU support
# model = model.cuda()
# model_size, factor, warmup, optimizer)
model_opt = NoamOpt(lr_model_sz, factor, warmup,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay))

loss_compute = SimpleLossCompute(Batch_Loss(trading_consumption, interest_rate, variance_penalty, cost_penalty, True),
                                 model_opt)
evaluate_loss_compute = SimpleLossCompute(
    Batch_Loss(trading_consumption, interest_rate, variance_penalty, cost_penalty, False), None)
test_loss_compute = SimpleLossCompute_tst(
    Test_Loss(trading_consumption, interest_rate, variance_penalty, cost_penalty, False), None)

##########################train net####################################################
tst_loss, tst_portfolio_value = train_net(DM, total_step, output_step, x_window_size, local_context_length, model,
                                          FLAGS.model_dir, FLAGS.model_index, loss_compute, evaluate_loss_compute, True,
                                          True)

model = torch.load(FLAGS.model_dir + '/' + str(FLAGS.model_index) + '.pkl')

##########################test net#####################################################
tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO = test_net(DM, 1, 1, x_window_size, local_context_length, model,
                                                               loss_compute, test_loss_compute, False, True)

csv_dir = FLAGS.log_dir + "/" + "train_summary.csv"
d = {"net_dir": [FLAGS.model_index],
     "fAPV": [tst_portfolio_value.item()],
     "SR": [SR.item()],
     "CR": [CR.item()],
     "TO": [TO.item()],
     "St_v": [''.join(str(e) + ', ' for e in St_v)],
     "backtest_test_history": [''.join(str(e) + ', ' for e in tst_pc_array.cpu().numpy())],
     }
new_data_frame = pd.DataFrame(data=d).set_index("net_dir")
if os.path.isfile(csv_dir):
    dataframe = pd.read_csv(csv_dir).set_index("net_dir")
    dataframe = dataframe.append(new_data_frame)
else:
    dataframe = new_data_frame
dataframe.to_csv(csv_dir)
