import datetime
import numpy as np
import torch
# from torch.autograd import Variable
import torch.nn as nn
from matplotlib import pyplot as plt
import Proximal_Adam


def SYM_Build(sizeY):
    SYM = np.zeros((sizeY * sizeY, sizeY))
    for i in range(sizeY - 1):
        for j in range(sizeY - i - 1):
            SYM[i * sizeY + j, i + j] += 1.0
            SYM[(sizeY - i - 1) * sizeY + sizeY - j - 1, j - i] += 1.0
    SYM[(sizeY - 1) * sizeY, sizeY - 1] += 1.0
    SYM[sizeY - 1, sizeY - 1] += 1.0
    return SYM


def M_Build(size_y):
    dM_dlambda = np.zeros((size_y * size_y, size_y))
    M = np.zeros((size_y * size_y, size_y))
    for i in range(size_y):
        for j in range(i):
            M[i * size_y + j, i - j] = 1.0
            dM_dlambda[i * size_y + j, i - j] = 1.0
        for j in range(i, size_y):
            M[i * size_y + j, j - i] = 1.0
            dM_dlambda[i * size_y + j, j - i] = 1.0
    return M, dM_dlambda


def lambdaSumMtc_Build(size_y):
    lambdaSumMtc = np.zeros((size_y, size_y))
    for i in range(size_y):
        for j in range(i):
            lambdaSumMtc[i, i - j] = lambdaSumMtc[i, i - j] + 1.0
        for j in range(i, size_y):
            lambdaSumMtc[i, j - i] = lambdaSumMtc[i, j - i] + 1.0
    return lambdaSumMtc


def plot_lambda_2(lambda1, lambda2):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(lambda1)
    plt.title('lambda_1')
    plt.subplot(2, 1, 2)
    plt.plot(lambda2)
    plt.title('lambda_2')
    plt.show()
    return

def prox_l1(x, gamma):
    return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - gamma)


class Pattern_Extract:
    def __init__(self, Y, lr_l=1e-4, lr_A_E=1e-2, rou_l1_lambda=1, rou_l1_A=0.02, rou_l2_E=1,
                 rou_row_sum=1, dtype=torch.float32, device=torch.device("cuda"), max_iteration=100,epsilon_lambda=1e-4):

        self.dtype = dtype
        self.device = device
        self.Y = Y
        self.size_y1 = self.Y.shape[0]
        self.size_y2 = self.Y.shape[1]
        # prepared matrix
        self.M1, self.dM_dlambda1 = M_Build(self.size_y1)
        self.M2, self.dM_dlambda2 = M_Build(self.size_y2)
        self.lambdaSumMtc1 = lambdaSumMtc_Build(self.size_y1)
        self.lambdaSumMtc2 = lambdaSumMtc_Build(self.size_y2)
        self.M1 = torch.tensor(self.M1, device=device, dtype=dtype, requires_grad=True)
        self.dM_dlambda1 = torch.tensor(self.dM_dlambda1, device=device, dtype=dtype, requires_grad=True)
        self.lambdaSumMtc1 = torch.tensor(self.lambdaSumMtc1, device=device, dtype=dtype, requires_grad=True)
        self.M2 = torch.tensor(self.M2, device=device, dtype=dtype, requires_grad=True)
        self.dM_dlambda2 = torch.tensor(self.dM_dlambda2, device=device, dtype=dtype, requires_grad=True)
        self.lambdaSumMtc2 = torch.tensor(self.lambdaSumMtc2, device=device, dtype=dtype, requires_grad=True)
        self.SYM1 = SYM_Build(self.size_y1)
        self.SYM1 = torch.tensor(self.SYM1, device=device, dtype=dtype, requires_grad=True)
        self.SYM2 = SYM_Build(self.size_y2)
        self.SYM2 = torch.tensor(self.SYM2, device=device, dtype=dtype, requires_grad=True)

        self.p1 = int(self.size_y1 * 0.1)
        self.p2 = int(self.size_y2 * 0.1)

        self.learning_rate_l = lr_l
        self.learning_rate_A_E = lr_A_E
        self.rou_l1_lambda = rou_l1_lambda
        self.rou_l1_A = rou_l1_A
        self.rou_row_sum = rou_row_sum
        self.rou_l2_E = rou_l2_E
        self.max_iteration = max_iteration

        self.loss_lambda1 = 1e9
        self.loss_lambda2 = 1e9
        self.min_loss_lambda1 = 1e9
        self.min_loss_lambda2 = 1e9
        self.loss_A = 1e9
        self.loss_E = 1e9
        self.lambda2_p = None
        self.lambda1_p = None
        self.Y_A_E = None
        self.Y_A_ET = None
        self.A = None
        self.YT = None
        self.Y_r = None
        self.E = None
        self.lambda2 = None
        self.lambda1 = None
        self.final_lambda2_p = None
        self.final_lambda1_p = None
        self.optimizer_A = None
        self.optimizer_lambda2 = None
        self.optimizer_lambda1 = None
        self.optimizer_E = None
        self.epsilon_lambda = epsilon_lambda

    def opti_iteration(self):
        lambda1_p_init = (1 / (self.size_y1 - 2 * self.p1)) * torch.ones((self.size_y1 - 2 * self.p1), 1).clone().to(
            self.device)
        lambda1_p_init = lambda1_p_init.to(self.device).to(self.dtype)
        lambda2_p_init = (1 / (self.size_y2 - 2 * self.p2)) * torch.ones((self.size_y2 - 2 * self.p2), 1).clone().to(
            self.device)
        lambda2_p_init = lambda2_p_init.to(self.device).to(self.dtype)
        epsilon_lambda1 = self.epsilon_lambda
        epsilon_lambda2 = self.epsilon_lambda
        self.lambda1_p = nn.Parameter(lambda1_p_init, requires_grad=True)
        self.lambda2_p = nn.Parameter(lambda2_p_init, requires_grad=True)
        self.A = torch.zeros(self.Y.shape).to(self.device)
        self.A = nn.Parameter(self.A, requires_grad=True)
        self.E = torch.zeros(self.Y.shape).to(self.device)
        self.E = nn.Parameter(self.E, requires_grad=True)
        self.Y = torch.tensor(self.Y, device=self.device, dtype=self.dtype, requires_grad=True)

        self.optimizer_lambda1 = Proximal_Adam.ProximalAdam([self.lambda1_p], lr=self.learning_rate_l)
        self.optimizer_lambda2 = Proximal_Adam.ProximalAdam([self.lambda2_p], lr=self.learning_rate_l)
        # self.optimizer_A = Proximal_Adam.ProximalAdam([self.A], lr=self.learning_rate_A_E)
        # self.optimizer_E = Proximal_Adam.ProximalAdam([self.E], lr=self.learning_rate_A_E)

        # self.optimizer_lambda1 = torch.optim.Adam([self.lambda1_p], lr=self.learning_rate_l)
        # self.optimizer_lambda2 = torch.optim.Adam([self.lambda2_p], lr=self.learning_rate_l)
        self.optimizer_A = torch.optim.Adam([self.A], lr=self.learning_rate_A_E)
        self.optimizer_E = torch.optim.Adam([self.E], lr=self.learning_rate_A_E)



        self.YT = (self.Y - torch.zeros(self.Y.shape).to(self.device)).transpose(0, 1)
        self.Y_A_E = (self.Y - self.A - self.E).detach()
        self.Y_A_ET = (self.Y - self.A - self.E).transpose(0, 1).detach()
        last_lambda1 = self.lambda1_p.clone()
        last_lambda2 = self.lambda2_p.clone()
        pre_lambda1 = self.lambda1_p.clone()
        pre_lambda2 = self.lambda2_p.clone()

        # first round optimization
        self.opti_lambda1()
        self.opti_lambda2()

        lambda_opti_time = 0

        while torch.norm(pre_lambda1 - self.lambda1_p, torch.inf) > epsilon_lambda1 or torch.norm(
                pre_lambda2 - self.lambda2_p, torch.inf) > epsilon_lambda2:
            lambda_opti_time+=1
            # opti lambda
            pre_lambda1 = self.lambda1_p.clone()
            pre_lambda2 = self.lambda2_p.clone()
            self.opti_lambda1()
            self.opti_lambda2()
        self.opti_A_iteration()
        self.opti_E_iteration()


        while (torch.norm(last_lambda1 - self.lambda1_p, torch.inf) > epsilon_lambda1 or torch.norm(
                last_lambda2 - self.lambda2_p,
                torch.inf) > epsilon_lambda2):

            last_lambda1 = self.lambda1_p.clone()
            last_lambda2 = self.lambda2_p.clone()
            # last_A = self.A.clone()
            pre_lambda1 = self.lambda1_p.clone()
            pre_lambda2 = self.lambda2_p.clone()
            lambda_opti_time += 1
            self.opti_lambda1()
            self.opti_lambda2()

            while torch.norm(pre_lambda1 - self.lambda1_p, torch.inf) > epsilon_lambda1 or torch.norm(
                    pre_lambda2 - self.lambda2_p, torch.inf) > epsilon_lambda2 or lambda_opti_time<300:
                # opti lambda
                lambda_opti_time += 1
                pre_lambda1 = self.lambda1_p.clone()
                pre_lambda2 = self.lambda2_p.clone()
                self.opti_lambda1()
                self.opti_lambda2()
            self.opti_A_iteration()
            self.opti_E_iteration()
            self.Y_A_E = (self.Y - self.A - self.E).detach()
            self.Y_A_ET = (self.Y - self.A - self.E).transpose(0, 1).detach()

        self.Y_A_E = (self.Y - self.A - self.E).cpu().detach().numpy()
        self.Y_A_ET = (self.Y - self.A - self.E).transpose(0, 1).detach()
        self.final_lambda1_p = self.lambda1_p.cpu().detach().numpy()
        self.final_lambda2_p = self.lambda2_p.cpu().detach().numpy()
        self.lambda1 = np.concatenate([np.zeros((self.p1, 1)), self.final_lambda1_p, np.zeros((self.p1, 1))])
        self.lambda2 = np.concatenate([np.zeros((self.p2, 1)), self.final_lambda2_p, np.zeros((self.p2, 1))])
        self.SYM1 = self.SYM1.cpu().detach().numpy()
        self.SYM2 = self.SYM2.cpu().detach().numpy()
        self.M1 = self.M1.cpu().detach().numpy()
        self.M2 = self.M2.cpu().detach().numpy()
        self.lambdaSumMtc1 = self.lambdaSumMtc1.cpu().detach().numpy()
        self.lambdaSumMtc2 = self.lambdaSumMtc2.cpu().detach().numpy()
        # M_lambda1 = np.dot(self.M1, self.lambda1).reshape(self.size_y1, self.size_y1)
        # M_lambda2 = np.dot(self.M2, self.lambda2).reshape(self.size_y2, self.size_y2)
        # w1 = np.diag((1 / np.dot(self.lambdaSumMtc1, self.lambda1)).reshape(self.size_y1))
        # w2 = np.diag((1 / np.dot(self.lambdaSumMtc2, self.lambda2)).reshape(self.size_y2))
        self.Y = self.Y.cpu().detach().numpy()
        self.A = self.A.cpu().detach().numpy()
        self.E = self.E.cpu().detach().numpy()
        # self.Y_r = np.transpose(
        #     np.dot(np.dot(w2, M_lambda2), np.transpose(np.dot(np.dot(w1, M_lambda1), self.Y - self.A)))) # 20240527
        self.Y_r = self.Y - self.A - self.E
        # self.Y_r = np.transpose(
        #     np.dot(M_lambda2, np.transpose(np.dot( M_lambda1, self.Y - self.A))))
        torch.cuda.empty_cache()
        return

    def opti_lambda1(self):
        lambda1 = torch.concatenate(
            [torch.zeros(self.p1, 1).to(self.device), self.lambda1_p, torch.zeros(self.p1, 1).to(self.device)])
        M_lambda1 = torch.mm(self.M1, lambda1).reshape(self.size_y1, self.size_y1)
        s_sum_1 = torch.mm(self.lambdaSumMtc1, lambda1.abs()).reshape(self.size_y1)
        loss1 = (torch.mm(M_lambda1, self.Y_A_E) - self.Y).pow(2).sum()
        loss2 = self.rou_row_sum * ((s_sum_1 - 1).pow(2).sum())
        loss3 = self.rou_l1_lambda * torch.abs(self.lambda1_p).sum()
        self.loss_lambda1 = loss1 + loss2 + loss3

        self.optimizer_lambda1.zero_grad()
        self.loss_lambda1.backward()
        def prox_func(params,gamma):
            params[0] = prox_l1(params[0], gamma)  # 对 A 应用 L1 正则化
            return params
        self.optimizer_lambda1.step(prox_func=prox_func)

        return

    def opti_lambda2(self):
        lambda2 = torch.concatenate(
            [torch.zeros(self.p2, 1).to(self.device), self.lambda2_p, torch.zeros(self.p2, 1).to(self.device)])
        M_lambda2 = torch.mm(self.M2, lambda2).reshape(self.size_y2, self.size_y2)
        s_sum_2 = torch.mm(self.lambdaSumMtc2, lambda2.abs()).reshape(self.size_y2)
        loss1 = (torch.mm(M_lambda2, self.Y_A_ET) - self.YT).pow(2).sum()
        loss2 = self.rou_row_sum * ((s_sum_2 - 1).pow(2).sum())
        loss3 = self.rou_l1_lambda * torch.abs(self.lambda2_p).sum()
        self.loss_lambda2 = loss1 + loss2 + loss3
        self.optimizer_lambda2.zero_grad()
        self.loss_lambda2.backward()
        def prox_func(params,gamma):
            params[0] = prox_l1(params[0], gamma)  # 对 A 应用 L1 正则化
            return params
        self.optimizer_lambda2.step(prox_func=prox_func)
        return

    def opti_A_iteration(self):
        lambda1 = torch.concatenate(
            [torch.zeros(self.p1, 1).to(self.device), self.lambda1_p, torch.zeros(self.p1, 1).to(self.device)])
        lambda2 = torch.concatenate(
            [torch.zeros(self.p2, 1).to(self.device), self.lambda2_p, torch.zeros(self.p2, 1).to(self.device)])
        M_lambda1 = torch.mm(self.M1, lambda1).reshape(self.size_y1, self.size_y1)
        M_lambda2 = torch.mm(self.M2, lambda2).reshape(self.size_y2, self.size_y2)
        w1 = torch.diag(1 / torch.mm(self.lambdaSumMtc1, lambda1).reshape(self.size_y1))
        w2 = torch.diag(1 / torch.mm(self.lambdaSumMtc2, lambda2).reshape(self.size_y2))
        RA1 = torch.mm(w1, M_lambda1).detach()
        RA2 = torch.mm(w2, M_lambda2).detach()
        for i in range(self.max_iteration):
            # loss
            loss1 = (torch.mm(RA2, torch.mm(RA1, self.Y - self.A-self.E).transpose(0, 1)) - (
                    self.Y - self.A-self.E).transpose(0, 1)).pow(2).sum()
            loss2 = self.rou_l1_A * torch.norm(self.A, 1)
            self.loss_A = loss1 + loss2
            self.optimizer_A.zero_grad()
            self.loss_A.backward()
            self.optimizer_A.step()
        return

    def opti_E_iteration(self):
        lambda1 = torch.concatenate(
            [torch.zeros(self.p1, 1).to(self.device), self.lambda1_p, torch.zeros(self.p1, 1).to(self.device)])
        lambda2 = torch.concatenate(
            [torch.zeros(self.p2, 1).to(self.device), self.lambda2_p, torch.zeros(self.p2, 1).to(self.device)])
        M_lambda1 = torch.mm(self.M1, lambda1).reshape(self.size_y1, self.size_y1)
        M_lambda2 = torch.mm(self.M2, lambda2).reshape(self.size_y2, self.size_y2)
        w1 = torch.diag(1 / torch.mm(self.lambdaSumMtc1, lambda1).reshape(self.size_y1))
        w2 = torch.diag(1 / torch.mm(self.lambdaSumMtc2, lambda2).reshape(self.size_y2))
        RE1 = torch.mm(w1, M_lambda1).detach()
        RE2 = torch.mm(w2, M_lambda2).detach()
        # st_test = datetime.datetime.now()

        # C = torch.flatten(self.Y - self.A - torch.mm(torch.mm(self.RE1, self.Y - self.A),self.RE2.transpose(0,1)))
        # X = torch.eye(self.size_y1*self.size_y2).to(self.device)-torch.kron(self.RE2,self.RE1)
        # E = torch.mm(torch.inverse(torch.mm(X.transpose(0,1),X)+torch.eye(self.size_y1*self.size_y2).to(self.device)),C)

        # E = torch.mm(torch.inverse((2*torch.eye(self.size_y1*self.size_y2).to(self.device)+torch.kron(torch.mm(self.RE2.transpose(0,1),self.RE2),torch.mm(self.RE1.transpose(0,1),self.RE1))
        #     -torch.kron(self.RE2.transpose(0,1),self.RE1.transpose(0,1))-torch.kron(self.RE2,self.RE1))),C)
        # self.E = E.reshape(self.size_y1,-1)
        for i in range(self.max_iteration):
            # loss
            loss1 = (torch.mm(RE2, torch.mm(RE1, self.Y - self.A - self.E).transpose(0, 1)) - (
                    self.Y - self.A - self.E).transpose(0, 1)).pow(2).sum()
            loss2 = self.rou_l2_E * self.E.pow(2).sum()
            self.loss_E = loss1 + loss2
            self.optimizer_E.zero_grad()
            self.loss_E.backward()
            self.optimizer_E.step()
        return
