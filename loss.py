import torch
import torch.nn as nn
import numpy as np

from pgportfolio.tools.indicator import max_drawdown


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y):
        loss, portfolio_value = self.criterion(x, y)
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss, portfolio_value


class SimpleLossCompute_tst:
    "A simple loss compute and train function."
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, x, y):
        loss, portfolio_value, rewards, SR, CR, tst_pc_array, TO = self.criterion(x, y)
        return loss, portfolio_value, rewards, SR, CR, tst_pc_array, TO


class Batch_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, device, gamma=0.1, beta=0.1, size_average=True):
        super(Batch_Loss, self).__init__()
        self.gamma = gamma  # variance penalty
        self.beta = beta
        self.size_average = size_average
        self.commission_ratio = commission_ratio
        self.interest_rate = interest_rate

        self.target_device = device

    def forward(self, w, y):  # w:[128,1,12]   y:[128,11,4]
        close_price_ratio = y[:, :, 0:1]
        close_price_ratio = close_price_ratio.to(self.target_device)   #   [128,11,1]
        # future close prise (including cash)
        ones = torch.ones(close_price_ratio.size()[0], 1, 1)
        ones = ones.to(self.target_device)

        # Add cache close price ratio?
        close_price_ratio = torch.cat([ones, close_price_ratio], 1)

        close_price_ratio = close_price_ratio.to(self.target_device)         #[128,11,1]cat[128,1,1]->[128,12,1]
        reward = torch.matmul(w, close_price_ratio)  # [128,1,1]

        close_price_ratio = close_price_ratio.view(close_price_ratio.size()[0], close_price_ratio.size()[2],
                                                   close_price_ratio.size()[1])  # [128,1,12]
        ###############################################################################################################
        element_reward = w * close_price_ratio
        interest = torch.zeros(element_reward.size(), dtype=torch.float)
        interest = interest.to(self.target_device)
        interest[element_reward < 0] = element_reward[element_reward < 0]
        interest = torch.sum(interest, 2).unsqueeze(2) * self.interest_rate  # [128,1,1]
        ###############################################################################################################
        future_omega = w * close_price_ratio / reward  # [128,1,12]
        wt = future_omega[:-1]  # [128,1,12]
        wt1 = w[1:]  # [128,1,12]
        pure_pc = 1 - torch.sum(torch.abs(wt - wt1), -1) * self.commission_ratio  # [128,1]
        pure_pc=pure_pc.to(self.target_device)
        ones = torch.ones([1, 1])
        ones = ones.to(self.target_device)
        pure_pc = torch.cat([ones, pure_pc], 0)
        pure_pc = pure_pc.view(pure_pc.size()[0], 1, pure_pc.size()[1])  # [128,1,1]

        cost_penalty = torch.sum(torch.abs(wt - wt1), -1)
        ################## Deduct transaction fee ##################
        reward = reward * pure_pc  # reward=pv_vector
        ################## Deduct loan interest ####################
        reward = reward + interest
        portfolio_value = torch.prod(reward, 0)
        batch_loss = -torch.log(reward)
        #####################variance_penalty##############################
        #        variance_penalty = ((torch.log(reward)-torch.log(reward).mean())**2).mean()
        if self.size_average:
            loss = batch_loss.mean()  # + self.gamma*variance_penalty + self.beta*cost_penalty.mean()
            return loss, portfolio_value[0][0]
        else:
            loss = batch_loss.mean()  # +self.gamma*variance_penalty + self.beta*cost_penalty.mean() #(dim=0)
            return loss, portfolio_value[0][0]


class Test_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, device, gamma=0.1, beta=0.1):
        super(Test_Loss, self).__init__()
        self.gamma = gamma  # variance penalty
        self.beta = beta
        self.commission_ratio = commission_ratio
        self.interest_rate = interest_rate

        self.target_device = device

    def forward(self, w, y):  # w:[128,10,1,12] y(128,10,11,4)
        close_price = y[:, :, :, 0:1]
        close_price = close_price.to(self.target_device)    #   [128,10,11,1]
        ones = torch.ones(close_price.size()[0], close_price.size()[1], 1, 1)
        ones = ones.to(self.target_device)
        close_price = torch.cat([ones, close_price], 2)  # [128,10,11,1]cat[128,10,1,1]->[128,10,12,1]
        close_price = close_price.to(self.target_device)
        rewards = torch.matmul(w, close_price)  # [128,10,1,12] * [128,10,12,1] ->[128,10,1,1]
        close_price = close_price.view(close_price.size()[0], close_price.size()[1], close_price.size()[3],
                                       close_price.size()[2])  # [128,10,12,1] -> [128,10,1,12]
        ##############################################################################
        element_reward = w * close_price
        interest = torch.zeros(element_reward.size(), dtype=torch.float)
        interest = interest.to(self.target_device)
        interest[element_reward < 0] = element_reward[element_reward < 0]
        #        print("interest:",interest.size(),interest,'\r\n')
        interest = torch.sum(interest, 3).unsqueeze(3) * self.interest_rate  # [128,10,1,1]
        ##############################################################################
        future_omega = w * close_price / rewards  # [128,10,1,12]*[128,10,1,12]/[128,10,1,1]
        wt = future_omega[:, :-1]  # [128, 9,1,12]
        wt1 = w[:, 1:]  # [128, 9,1,12]
        pure_pc = 1 - torch.sum(torch.abs(wt - wt1), -1) * self.commission_ratio  # [128,9,1]
        pure_pc=pure_pc.to(self.target_device)
        ones = torch.ones([pure_pc.size()[0], 1, 1])
        ones = ones.to(self.target_device)
        pure_pc = torch.cat([ones, pure_pc], 1)  # [128,1,1] cat  [128,9,1] ->[128,10,1]
        pure_pc = pure_pc.view(pure_pc.size()[0], pure_pc.size()[1], 1,
                               pure_pc.size()[2])  # [128,10,1] ->[128,10,1,1]
        cost_penalty = torch.sum(torch.abs(wt - wt1), -1)  # [128, 9, 1]
        ################## Deduct transaction fee ##################
        rewards = rewards * pure_pc  # [128,10,1,1]*[128,10,1,1]  test: [1,2808-31,1,1]
        ################## Deduct loan interest ####################
        rewards = rewards + interest

        # Difference from train loss start {
        tst_pc_array = rewards.squeeze()
        sr_reward = tst_pc_array - 1
        SR = sr_reward.mean() / sr_reward.std()
        #            print("SR:",SR.size(),"reward.mean():",reward.mean(),"reward.std():",reward.std())
        SN = torch.prod(rewards, 1)  # [1,1,1,1]
        SN = SN.squeeze()  #
        #            print("SN:",SN.size())
        MDD = max_drawdown(tst_pc_array)
        CR = SN / MDD
        TO = cost_penalty.mean()
        # } Difference from train loss end


        ##############################################
        portfolio_value_history = np.cumprod(rewards, axis=1).squeeze()
        batch_loss = -torch.log(portfolio_value_history[-1])  # [128,1,1]

        loss = batch_loss.mean()
        return loss, portfolio_value_history, rewards.squeeze(), SR, CR, tst_pc_array, TO
