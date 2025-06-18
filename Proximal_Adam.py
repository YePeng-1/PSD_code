import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import math

class ProximalAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(ProximalAdam, self).__init__(params, defaults)

    def step(self, prox_func=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ProximalAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if prox_func:
                    # Proximal gradient update
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                    # Proximal sub-iterations
                    z = p.data.clone()
                    gamma = group['lr'] / torch.max(denom)
                    for tau in range(1, 100 + 1):  # prox_max_iter set to 100
                        z_ = prox_func(z - gamma / step_size * (z - p.data), gamma)
                        converged = torch.norm(z_ - z) <= 1e-6 * torch.norm(z)
                        z = z_
                        if converged:
                            break
                    p.data = z

                else:
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss