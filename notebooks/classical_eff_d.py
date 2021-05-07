import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable

def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)



# Import Data

def fisher(y, x):
    """

    :param y: function y = y(x) for which the gradient is calculated
    :param x: parameters which y(x) depends on
    :return: Fisher information matrix
    """
    j = jacobian(y, x, create_graph=True)
    j = j.detach().numpy()
    return torch.from_numpy(np.outer(j, j)).type(torch.DoubleTensor)


def normalise_fishers(Fishers, thetas, d, V):
    num_samples = len(Fishers)

    TrF_integral = (1 / num_samples) * torch.sum(torch.tensor([torch.trace(F) for F in Fishers]))
    return [((d * V) / TrF_integral) * F for F in Fishers]

class Model42(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4,2,bias=False)
    def forward(self,x):
        x = self.l1(x)
        return x
    def

class Model4112(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4,1,bias=False)
        self.l2 = nn.Linear(1, 1, bias=False)
        self.l2 = nn.Linear(1, 2, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        x = self.l1(x)
        x = self.sig(x)
        x = self.l2(x)
        x = self.sig(x)
        x = self.l3(x)
        return x


