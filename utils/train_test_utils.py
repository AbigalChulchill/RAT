from datetime import datetime

import torch
import time
import numpy as np
import pandas as pd


def train_one_step(DM, x_window_size, model, loss_compute, local_context_length, device):
    batch = DM.next_batch()
    batch_input = batch["X"]  # (128, 4, 11, 31)
    batch_y = batch["y"]  # (128, 4, 11)
    batch_last_w = batch["last_w"]  # (128, 11)
    batch_w = batch["setw"]
    #############################################################################
    previous_w = torch.tensor(batch_last_w, dtype=torch.float)
    previous_w = previous_w.to(device)
    previous_w = torch.unsqueeze(previous_w, 1)  # [128, 11] -> [128,1,11]
    batch_input = batch_input.transpose((1, 0, 2, 3))
    batch_input = batch_input.transpose((0, 1, 3, 2))
    src = torch.tensor(batch_input, dtype=torch.float)
    src = src.to(device)
    price_series_mask = (torch.ones(src.size()[1], 1, x_window_size) == 1)  # [128, 1, 31]
    currt_price = src.permute((3, 1, 2, 0))  # [4,128,31,11]->[11,128,31,4]
    if (local_context_length > 1):
        padding_price = currt_price[:, :, -(local_context_length) * 2 + 1:-1, :]
    else:
        padding_price = None
    currt_price = currt_price[:, :, -1:, :]  # [11,128,31,4]->[11,128,1,4]
    trg_mask = make_std_mask(currt_price, src.size()[1])
    batch_y = batch_y.transpose((0, 2, 1))  # [128, 4, 11] ->#[128,11,4]
    trg_y = torch.tensor(batch_y, dtype=torch.float)
    trg_y = trg_y.to(device)
    out = model.forward(src, currt_price, previous_w,
                        price_series_mask, trg_mask, padding_price)
    new_w = out[:, :, 1:]  # 去掉cash
    new_w = new_w[:, 0, :]  # #[109,1,11]->#[109,11]
    new_w = new_w.detach().cpu().numpy()
    batch_w(new_w)

    loss, portfolio_value = loss_compute(out, trg_y)
    return loss, portfolio_value


def test_online(DM, x_window_size, model, evaluate_loss_compute, local_context_length, device):
    tst_batch = DM.get_test_set_online(DM._test_ind[0], DM._test_ind[-1], x_window_size)
    tst_batch_input = tst_batch["X"]
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w = torch.tensor(tst_batch_last_w, dtype=torch.float)
    tst_previous_w = tst_previous_w.to(device)
    tst_previous_w = torch.unsqueeze(tst_previous_w, 1)

    tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
    tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))

    long_term_tst_src = torch.tensor(tst_batch_input, dtype=torch.float)
    long_term_tst_src = long_term_tst_src.to(device)
    #########################################################################################
    tst_src_mask = (torch.ones(long_term_tst_src.size()[1], 1, x_window_size) == 1)

    long_term_tst_currt_price = long_term_tst_src.permute((3, 1, 2, 0))
    long_term_tst_currt_price = long_term_tst_currt_price[:, :, x_window_size - 1:, :]
    ###############################################################################################
    tst_trg_mask = make_std_mask(long_term_tst_currt_price[:, :, 0:1, :], long_term_tst_src.size()[1])

    tst_batch_y = tst_batch_y.transpose((0, 3, 2, 1))
    tst_trg_y = torch.tensor(tst_batch_y, dtype=torch.float)
    tst_trg_y = tst_trg_y.to(device)
    tst_long_term_w = []
    tst_y_window_size = len(DM._test_ind) - x_window_size - 1 - 1
    for j in range(tst_y_window_size + 1):  # 0-9
        tst_src = long_term_tst_src[:, :, j:j + x_window_size, :]
        tst_currt_price = long_term_tst_currt_price[:, :, j:j + 1, :]
        if (local_context_length > 1):
            padding_price = long_term_tst_src[:, :,
                            j + x_window_size - 1 - local_context_length * 2 + 2:j + x_window_size - 1, :]
            padding_price = padding_price.permute((3, 1, 2, 0))  # [4, 1, 2, 11] ->[11,1,2,4]
        else:
            padding_price = None
        out = model.forward(tst_src, tst_currt_price, tst_previous_w,
                            # [109,1,11]   [109, 11, 31, 3]) torch.Size([109, 11, 3]
                            tst_src_mask, tst_trg_mask, padding_price)
        if (j == 0):
            tst_long_term_w = out.unsqueeze(0)  # [1,109,1,12]
        else:
            tst_long_term_w = torch.cat([tst_long_term_w, out.unsqueeze(0)], 0)
        out = out[:, :, 1:]  # 去掉cash #[109,1,11]
        tst_previous_w = out
    tst_long_term_w = tst_long_term_w.permute(1, 0, 2, 3)  ##[10,128,1,12]->#[128,10,1,12]
    tst_loss, portfolio_value_history, rewards, SR, CR, tst_pc_array, TO = evaluate_loss_compute(tst_long_term_w, tst_trg_y)
    return tst_loss, portfolio_value_history, rewards, SR, CR, tst_pc_array, TO, tst_long_term_w, tst_trg_y


def test_net(DM, total_step, output_step, x_window_size, local_context_length, model,
             evaluate_loss_compute, device, evaluate=True):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    ####每个epoch开始时previous_w=0

    #########################################################tst########################################################
    with torch.no_grad():
        model.eval()
        tst_loss, portfolio_value_history, rewards, SR, CR, \
        tst_pc_array, TO, tst_long_term_w, tst_trg_y = test_online(
            DM, x_window_size, model, evaluate_loss_compute, local_context_length, device)
        elapsed = time.time() - start
        print("Test Loss: %f| Portfolio_Value: %f | SR: %f | CR: %f | TO: %f |testset per Sec: %f" %
              (tst_loss.item(), portfolio_value_history[-1].item(), SR.item(), CR.item(), TO.item(), 1 / elapsed))
        start = time.time()
        #                portfolio_value_list.append(portfolio_value.item())

        log_SR = SR
        log_CR = CR
        log_tst_pc_array = tst_pc_array
    return portfolio_value_history, rewards, log_SR, log_CR, log_tst_pc_array, TO, tst_long_term_w, tst_trg_y


def test_episode(DM, x_window_size, model, evaluate_loss_compute, local_context_length, device):
    test_set = DM.get_validation_set()
    test_set_input = test_set["X"]  # (TEST_SET_SIZE, 4, 11, 31)
    test_set_y = test_set["y"]

    test_previous_w = torch.zeros([1, 1, test_set_input.shape[2]])

    losses = []
    portfolio_values = []

    for i in range(len(test_set_input)):
        tst_batch_input = test_set_input[i:i + 1]  # (1, 4, 11, 31)

        test_batch_y = test_set_y[i:i + 1]

        test_previous_w = test_previous_w.to(device)
        # test_previous_w = torch.unsqueeze(test_previous_w, 1)  # [2426, 1, 11]
        tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
        tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))
        tst_src = torch.tensor(tst_batch_input, dtype=torch.float)
        tst_src = tst_src.to(device)
        tst_src_mask = (torch.ones(tst_src.size()[1], 1, x_window_size) == 1)  # [128, 1, 31]
        tst_currt_price = tst_src.permute((3, 1, 2, 0))  # (4,128,31,11)->(11,128,31,3)
        #############################################################################
        if (local_context_length > 1):
            padding_price = tst_currt_price[:, :, -(local_context_length) * 2 + 1:-1, :]  # (11,128,8,4)
        else:
            padding_price = None
        #########################################################################

        tst_currt_price = tst_currt_price[:, :, -1:, :]  # (11,128,31,4)->(11,128,1,4)
        tst_trg_mask = make_std_mask(tst_currt_price, tst_src.size()[1])
        test_batch_y = test_batch_y.transpose((0, 2, 1))  # (128, 4, 11) ->(128,11,4)
        tst_trg_y = torch.tensor(test_batch_y, dtype=torch.float)
        tst_trg_y = tst_trg_y.to(device)
        ###########################################################################################################
        tst_out = model.forward(tst_src, tst_currt_price, test_previous_w,  # [128,1,11]   [128, 11, 31, 4])
                                tst_src_mask, tst_trg_mask, padding_price)

        tst_loss, tst_portfolio_value = evaluate_loss_compute(tst_out, tst_trg_y)

        tst_loss = tst_loss.item()
        losses.append(tst_loss)
        portfolio_values.append(tst_portfolio_value)

        # exclude the cash
        test_previous_w = tst_out[:, :, 1:]

    return np.mean(losses), np.prod(portfolio_values)


def test_batch(DM, x_window_size, model, evaluate_loss_compute, local_context_length, device):
    tst_batch = DM.get_validation_set()
    tst_batch_input = tst_batch["X"]  # (128, 4, 11, 31)
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w = torch.tensor(tst_batch_last_w, dtype=torch.float)
    tst_previous_w = tst_previous_w.to(device)
    tst_previous_w = torch.unsqueeze(tst_previous_w, 1)  # [2426, 1, 11]

    tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
    tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))

    tst_src = torch.tensor(tst_batch_input, dtype=torch.float)
    tst_src = tst_src.to(device)

    tst_src_mask = (torch.ones(tst_src.size()[1], 1, x_window_size) == 1)  # [128, 1, 31]

    tst_currt_price = tst_src.permute((3, 1, 2, 0))  # (4,128,31,11)->(11,128,31,3)
    #############################################################################
    if (local_context_length > 1):
        padding_price = tst_currt_price[:, :, -(local_context_length) * 2 + 1:-1, :]  # (11,128,8,4)
    else:
        padding_price = None
    #########################################################################

    tst_currt_price = tst_currt_price[:, :, -1:, :]  # (11,128,31,4)->(11,128,1,4)
    tst_trg_mask = make_std_mask(tst_currt_price, tst_src.size()[1])
    tst_batch_y = tst_batch_y.transpose((0, 2, 1))  # (128, 4, 11) ->(128,11,4)
    tst_trg_y = torch.tensor(tst_batch_y, dtype=torch.float)
    tst_trg_y = tst_trg_y.to(device)
    ###########################################################################################################
    tst_out = model.forward(tst_src, tst_currt_price, tst_previous_w,  # [128,1,11]   [128, 11, 31, 4])
                            tst_src_mask, tst_trg_mask, padding_price)

    tst_loss, tst_portfolio_value = evaluate_loss_compute(tst_out, tst_trg_y)
    return tst_loss, tst_portfolio_value


def train_net(DM, total_step, output_step, x_window_size, local_context_length, model, output_dir,
              loss_compute, evaluate_loss_compute, device):
    "Standard Training and Logging Function"
    start = time.time()
    # total_loss = 0
    ####每个epoch开始时previous_w=0
    max_tst_portfolio_value = 0

    log = []

    for i in range(total_step):
        model.train()
        loss, portfolio_value = train_one_step(DM, x_window_size, model, loss_compute, local_context_length, device)
        # total_loss += loss.item()
        if (i % output_step == 0):
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                  (i, loss.item(), portfolio_value.item(), output_step / elapsed))
            start = time.time()
            #########################################################tst########################################################
            with torch.no_grad():
                model.eval()
                tst_loss, tst_portfolio_value = test_batch(DM, x_window_size, model, evaluate_loss_compute,
                                                           local_context_length, device)
                elapsed = time.time() - start
                print("Test: %d Loss: %f| Portfolio_Value: %f | testset per Sec: %f \r\n" %
                      (i, tst_loss.item(), tst_portfolio_value.item(), 1 / elapsed))
                start = time.time()

                log.append({
                    "time": datetime.now().isoformat(sep=' ', timespec='seconds'),
                    "epoch": i + 1,
                    "train_loss": loss.item(),
                    "train_apv": portfolio_value.item(),
                    "test_loss": tst_loss.item(),
                    "test_apv": tst_portfolio_value.item()
                })

                pd.DataFrame(log).to_csv(f"{output_dir}/train_log.csv", index=False)

                if tst_portfolio_value > max_tst_portfolio_value:
                    max_tst_portfolio_value = tst_portfolio_value
                    torch.save(model, f"{output_dir}/best_model.pkl")
                    print("save model!")

    return tst_loss, tst_portfolio_value


def make_std_mask(local_price_context, batch_size):
    "Create a mask to hide padding and future words."
    local_price_mask = (torch.ones(batch_size, 1, 1) == 1)
    local_price_mask = local_price_mask & (subsequent_mask(local_price_context.size(-2)).type_as(local_price_mask.data))
    return local_price_mask


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
