import os

import pandas as pd
import torch

from loss import SimpleLossCompute_tst, Test_Loss
from parameters_stocks import FLAGS, DM
from pgportfolio.marketdata.datamatricesnew import DataMatricesNew
from utils.train_test_utils import test_net

test_model_index = 1

run_dir = "./output/20220419_233859"

model = torch.load(run_dir + '/best_model.pkl')

device = "cuda"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"S


model = model.to(device)

##########################test net#####################################################
test_loss_compute = SimpleLossCompute_tst(
    Test_Loss(FLAGS.trading_consumption, FLAGS.interest_rate, device, FLAGS.variance_penalty, FLAGS.cost_penalty, False))

tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO = test_net(
    DM, 1, 1, FLAGS.x_window_size, FLAGS.local_context_length, model,
    test_loss_compute, device, False)

csv_dir = f"{run_dir}/test_summary.csv"
d = {"net_dir": [test_model_index],
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
