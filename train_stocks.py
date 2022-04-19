import os

import torch

from loss import SimpleLossCompute, Batch_Loss
from parameters_stocks import device, FLAGS, DM
from rat.rat import make_model
from utils.train_test_utils import train_net


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

os.makedirs(FLAGS.log_dir, exist_ok=True)
os.makedirs(FLAGS.model_dir, exist_ok=True)




coin_num = len(DM.global_matrix.coins)  # 11


print(f"Torch device is : {device}")

model = make_model(FLAGS.batch_size, coin_num, FLAGS.x_window_size, FLAGS.feature_number,
                   N=1, d_model_Encoder=FLAGS.multihead_num * FLAGS.model_dim,
                   d_model_Decoder=FLAGS.multihead_num * FLAGS.model_dim,
                   d_ff_Encoder=FLAGS.multihead_num * FLAGS.model_dim,
                   d_ff_Decoder=FLAGS.multihead_num * FLAGS.model_dim,
                   h=FLAGS.multihead_num,
                   dropout=0.01,
                   local_context_length=FLAGS.local_context_length)

model = model.to(device)

# model = make_model3(N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
# model_size, factor, warmup, optimizer)

#################set learning rate###################
lr_model_sz = 5120
factor = FLAGS.learning_rate  # 1.0
warmup = 0  # 800


model_opt = NoamOpt(lr_model_sz, factor, warmup,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=FLAGS.weight_decay))

loss_compute = SimpleLossCompute(
    Batch_Loss(FLAGS.trading_consumption, FLAGS.interest_rate, device, FLAGS.variance_penalty, FLAGS.cost_penalty, True),
    model_opt)
evaluate_loss_compute = SimpleLossCompute(
    Batch_Loss(FLAGS.trading_consumption, FLAGS.interest_rate, device, FLAGS.variance_penalty, FLAGS.cost_penalty, False), None)

##########################train net####################################################
tst_loss, tst_portfolio_value = train_net(DM, FLAGS.total_step, FLAGS.output_step, FLAGS.x_window_size, FLAGS.local_context_length, model,
                                          FLAGS.model_dir, FLAGS.model_index, loss_compute, evaluate_loss_compute,
                                          device, True, True)
