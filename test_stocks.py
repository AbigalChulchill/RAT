import os
import numpy as np
import pandas as pd
import torch

from loss import SimpleLossCompute_tst, Test_Loss
from parameters_stocks import FLAGS, DM
from pgportfolio.marketdata.datamatricesnew import DataMatricesNew
from utils.train_test_utils import test_net

test_model_index = 1

run_dir = "./output/20220419_233953"

model = torch.load(run_dir + '/best_model.pkl')

device = "cpu"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"S


model = model.to(device)

##########################test net#####################################################
test_loss_compute = SimpleLossCompute_tst(
    Test_Loss(FLAGS.trading_consumption, FLAGS.interest_rate, device, FLAGS.variance_penalty, FLAGS.cost_penalty))

portfolio_value_history, rewards, SR, CR, tst_pc_array, TO, tst_long_term_w, tst_trg_y = test_net(
    DM, 1, 1, FLAGS.x_window_size, FLAGS.local_context_length, model,
    test_loss_compute, device, False)

asset_names = DM.global_matrix.coins.to_series().str.lower()
test_period_time = DM.global_matrix.time[
                   DM._test_ind[0] + FLAGS.x_window_size + 1:DM._test_ind[-1] + 1].to_series().tolist()

portfolio_distribution = pd.DataFrame(
    tst_long_term_w.reshape(-1, len(DM.global_matrix.coins) + 1),
    columns=['asset_cash'] + ('asset_' + asset_names).tolist(),
    index=test_period_time)

price_change_df = pd.DataFrame(tst_trg_y[..., 0].reshape(-1, len(DM.global_matrix.coins)),
                               columns='price_change_' + asset_names,
                               index=test_period_time)

test_results = portfolio_distribution.assign(rewards=rewards,
                                             portfolio_value=portfolio_value_history)

test_results = test_results.join(price_change_df)

csv_dir = f"{run_dir}/test_summary.csv"
d = {"net_dir": [test_model_index],
     "fAPV": [portfolio_value_history[-1].item()],
     "SR": [SR.item()],
     "CR": [CR.item()],
     "TO": [TO.item()],
     "St_v": [''.join(str(e) + ', ' for e in portfolio_value_history)],
     "backtest_test_history": [''.join(str(e) + ', ' for e in tst_pc_array.cpu().numpy())],
     }
new_data_frame = pd.DataFrame(data=d).set_index("net_dir")
if os.path.isfile(csv_dir):
    dataframe = pd.read_csv(csv_dir).set_index("net_dir")
    dataframe = dataframe.append(new_data_frame)
else:
    dataframe = new_data_frame
dataframe.to_csv(csv_dir)
