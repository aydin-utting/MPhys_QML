import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from ast import literal_eval
#from ..models.fisher_functions import fisher
from tqdm.auto import tqdm

# FUNCTIONS


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




def mathematica_import(s):
    s = s.replace('{', '[')
    s = s.replace('}', ']')
    l = literal_eval(s)
    return l


# MATHEMATICA IMPORT

# 2 HIDDEN LAYER
RAW_H2 = "{{2, 8}, {4, 4}, {6, 2}}"
H2 = mathematica_import(RAW_H2)
# 3 HIDDEN LAYERS
RAW_H3 = "{{2, 2, 7}, {2, 4, 4}, {2, 7, 2}, {4, 2, 4}}"
H3 = mathematica_import(RAW_H3)
# 4 HIDDEN LAYERS
RAW_H4 = "{{2, 2, 2, 6}, {2, 2, 6, 2}, {2, 3, 2, 5}, {2, 4, 2, 4}, {2, 5, 2, 3}, {2, 6, 2, 2}, {3, 3, 3, 2}, {3, 4, 2, 2}, {4, 2, 2, 3}, {4, 2, 3, 2}}"
H4 = mathematica_import(RAW_H4)
#5 HIDDEN LAYERS
RAW_H5 = "{{2, 2, 2, 2, 5}, {2, 2, 2, 5, 2}, {2, 2, 3, 2, 4}, {2, 2, 4, 2, 3}, {2, 2, 5, 2, 2}, {2, 3, 2, 2, 4}, {2, 3, 2, 4, 2}, {2, 4, 2, 2, 3}, {2, 4, 2, 3, 2}, {2, 5, 2, 2, 2}, {4, 2, 2, 2, 2}}"
H5 = mathematica_import(RAW_H5)
#6 HIDDEN LAYERS
RAW_H6 = "{{2, 2, 2, 2, 2, 4}, {2, 2, 2, 2, 4, 2}, {2, 2, 2, 3, 2, 3}, {2, 2, 2,4, 2, 2}, {2, 2, 3, 2, 2, 3}, {2, 2, 3, 2, 3, 2}, {2, 2, 4, 2, 2, 2}, {2, 3, 2, 2, 2, 3}, {2, 3, 2, 2, 3, 2}, {2, 3, 2, 3, 2, 2}, {2, 4, 2, 2, 2, 2}}"
H6 = mathematica_import(RAW_H6)
#7 HIDDEN LAYERS
RAW_H7 = "{{2, 2, 2, 2, 2, 2, 3}, {2, 2, 2, 2, 2, 3, 2}, {2, 2, 2, 2, 3, 2, 2}, {2, 2, 2, 3, 2, 2, 2}, {2, 2, 3, 2, 2, 2, 2}, {2, 3, 2, 2, 2, 2,2}}"
H7 = mathematica_import(RAW_H7)
#8 HIDDEN LAYERS
RAW_H8 = "{{2, 2, 2, 2, 2, 2, 2, 2}}"
H8 = mathematica_import(RAW_H8)


"""
#IMPORT IRIS DATA
df = pd.read_csv('../data/iris.data', names=['a', 'b', 'c', 'd', 'species'])
df = df.query("species=='Iris-setosa' or species=='Iris-versicolor'")
df.replace('Iris-setosa', 0, inplace=True)
df.replace('Iris-versicolor', 1, inplace=True)
X = Variable(torch.tensor(df.astype(float).values[:, 0:4]).float())
Y = Variable(torch.tensor(df.values.astype(int)[:, 4]).long())
"""

#CREATE GAUSSIAN DATA
def create_data(n_samples):

    X0 = torch.tensor(torch.tensor([[torch.normal(torch.tensor([1.]),torch.tensor([1.])),
                                 torch.normal(torch.tensor([-1.]),torch.tensor([1.])),
                                 torch.normal(torch.tensor([1.]),torch.tensor([1.])),
                                 torch.normal(torch.tensor([-1.]),torch.tensor([1.]))] for i in range(n_samples // 2)]),
                      requires_grad=False)

    X1 = torch.tensor(torch.tensor([[torch.normal(torch.tensor([-1.]), torch.tensor([1.])),
                                 torch.normal(torch.tensor([1.]), torch.tensor([1.])),
                                 torch.normal(torch.tensor([-1.]), torch.tensor([1.])),
                                 torch.normal(torch.tensor([1.]), torch.tensor([1.]))] for i in
                                range(n_samples // 2)]),
                      requires_grad=False)

    X = torch.cat((X0, X1), 0)

    Y = torch.cat((torch.tensor([torch.tensor([0]) for i in range(n_samples // 2)]),
                   torch.tensor([torch.tensor([1]) for i in range(n_samples // 2)])), 0)

    return X,Y


class model:
    def __init__(self, sizes, s_in, s_out):
        self.sizes = [s_in] + sizes + [s_out]

    def __call__(self, x, w):
        pos = 0
        for i, n in enumerate(self.sizes[:-1]):
            x = F.linear(x, w[pos:pos + self.sizes[i] * self.sizes[i + 1]].view(self.sizes[i + 1], self.sizes[i]))
        return x

X,Y = create_data(1000)
repeats = 10
sets = H2 + H3 + H4 + H5 + H6 + H7 + H8
ranks = np.empty((repeats,len(sets)))
for r in tqdm(range(repeats)):
    for j,sizes in enumerate(sets):
        w = torch.rand((40,), requires_grad=True).float()
        net = model(sizes,4,2)
        pred = net(X, w)
        loss = F.cross_entropy(pred,Y)
        Fisher = torch.zeros((40,40))
        for x, y in zip(X, Y):
            pred = F.softmax(net(x.view(1, 4),w),dim=1)
            loss = F.cross_entropy(pred, y.view(1, ))
            Fisher = Fisher + fisher(loss, w)
        Fisher = Fisher / len(X)
        ranks[r, j]=torch.matrix_rank(Fisher)

max_arg = ranks.mean(axis=0).argmax()
max_rank_config = sets[max_arg]
print(f"The configuration with max rank is: {max_rank_config}")


