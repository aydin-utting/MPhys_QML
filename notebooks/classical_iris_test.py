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

# Import Data
df = pd.read_csv('../data/iris.data',names=['a','b','c','d','species'])
df=df.query("species=='Iris-setosa' or species=='Iris-versicolor'")
df.replace('Iris-setosa',0,inplace=True)
df.replace('Iris-versicolor',1,inplace=True)
X = Variable(torch.tensor(df.astype(float).values[:,0:4]).float())
Y = Variable(torch.tensor(df.values.astype(int)[:,4]).long())


# TRAIN
runs = 50
epochs = 100
epoch_loss=np.empty((runs,epochs))

for r in range(runs):
    net = nn.Linear(4, 2, bias=False)
    nn.init.uniform_(net.weight, -1., 1.)
    opt = torch.optim.Adam(net.parameters(), lr=0.1)
    for epoch in range(epochs):
        opt.zero_grad()
        pred = net(X)
        pred = F.softmax(pred)
        loss = F.cross_entropy(pred,Y)
        loss.backward()
        opt.step()
        epoch_loss[r,epoch]=loss.item()

    #Fisher info
    Fisher = torch.zeros((8,8),requires_grad=True)
    w = net.weight.view(8, ).double()
    for x,y in zip(X,Y):
        pred=F.softmax(net(x.view(1,4)))
        loss = F.cross_entropy(pred,y.view(1,))
        Fisher=Fisher+fisher(loss,net.weight)
    Fisher=Fisher/len(X)
    FR = torch.matmul(w,torch.matmul(Fisher,w))
    print(f'Fisher-Rao Norm: {FR.item()}, Final Loss: {epoch_loss[r,-1]:2f} ')
plt.plot(range(epochs),epoch_loss.mean(axis=0))
plt.fill_between(range(epochs),epoch_loss.mean(axis=0)-epoch_loss.std(axis=0),epoch_loss.mean(axis=0)+epoch_loss.std(axis=0),alpha=0.2)
plt.show()

plt.plot(range(epochs),epoch_loss.transpose())
plt.show()
