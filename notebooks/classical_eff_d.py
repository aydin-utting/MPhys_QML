#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from tqdm.auto import tqdm

DIM=8

def initialise_data(n_samples):
    variance = 1
    X0 = np.array([[np.random.normal(loc=-1, scale=variance),
                    np.random.normal(loc=1, scale=variance),
                    np.random.normal(loc=-1, scale=variance),
                    np.random.normal(loc=1, scale=variance)] for i in range(int(n_samples / 2))])

    X1 = np.array([[np.random.normal(loc=1, scale=variance),
                    np.random.normal(loc=-1, scale=variance),
                    np.random.normal(loc=1, scale=variance),
                    np.random.normal(loc=-1, scale=variance)] for i in range(int(n_samples / 2))])

    X = np.append(X0, X1, 0)

    Y = np.append(np.array([0 for i in range(int(n_samples / 2))]),
                  np.array([1 for i in range(int(n_samples / 2))]))

    data = list(zip(X, Y))
    X = Variable(torch.tensor(X).float())
    Y = Variable(torch.tensor(Y))
    return X,Y

# Import Data
def get_iris():
    df = pd.read_csv('../data/iris.data',names=['a','b','c','d','species'])
    df=df.query("species=='Iris-setosa' or species=='Iris-versicolor'")
    df.replace('Iris-setosa',0,inplace=True)
    df.replace('Iris-versicolor',1,inplace=True)
    X = Variable(torch.tensor(df.astype(float).values[:,0:4]).float())
    Y = Variable(torch.tensor(df.values.astype(int)[:,4]).long())
    return X,Y

def jacobian(y, x, create_graph=False):
    num_x = 0
    for p in x:
        num_x += p.numel()

    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(num_x))
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
    return np.outer(j, j)


def normalise_fishers(Fishers):
    TrF_integral = Fishers.trace(axis1=1, axis2=2).mean()
    return (DIM / TrF_integral) * Fishers


def effective_dimension(normed_fishers, n_data: int):
    id = torch.eye(DIM)
    kappa = n_data / (2 * np.pi * np.log(n_data))
    integral = np.average(np.sqrt(np.linalg.det(id + kappa * normed_fishers)))
    numerator = np.log(integral)
    return 2 * numerator / np.log(kappa)


class Net42(nn.Module):
    def __init__(self):
        W =(torch.rand(8)-0.5)*2.
        super().__init__()
        self.W = nn.Parameter(W)
    def forward(self,x):
        w1 = self.W.view((2,4))
        x = F.linear(x,w1)
        return x
    def get_W(self):
        return self.W

class Net4112(nn.Module):
    def __init__(self):
        W =(torch.rand(8)-0.5)*2.
        super().__init__()
        self.W = nn.Parameter(W)
    def forward(self,x):
        w1 = self.W[0:4].view((1,4))
        x = F.sigmoid(F.linear(x,w1))
        w2 = self.W[4:5].view((1, 1))
        x = F.sigmoid(F.linear(x, w2))
        w3 = self.W[5:6].view((1, 1))
        x = F.sigmoid(F.linear(x, w3))
        w4 = self.W[6:8].view((2, 1))
        x = F.linear(x, w4)
        return x

    def get_W(self):
        return self.W

class Net4622(nn.Module):
    def __init__(self):
        W =(torch.rand(40)-0.5)*2.
        super().__init__()
        self.W = nn.Parameter(W)
    def forward(self,x):
        w1 = self.W[0:24].view((6,4))
        x = F.sigmoid(F.linear(x,w1))
        w2 = self.W[24:36].view((2, 6))
        x = F.sigmoid(F.linear(x, w2))
        w4 = self.W[36:40].view((2, 2))
        x = F.linear(x, w4)
        return x
    def get_W(self):
        return self.W

class Net4442(nn.Module):
    def __init__(self):
        W =(torch.rand(40)-0.5)*2.
        super().__init__()
        self.W = nn.Parameter(W)
    def forward(self,x):
        w1 = self.W[0:16].view((4,4))
        x = F.sigmoid(F.linear(x,w1))
        w2 = self.W[16:32].view((4, 4))
        x = F.sigmoid(F.linear(x, w2))
        w4 = self.W[32:40].view((2, 4))
        x = F.linear(x, w4)
        return x
    def get_W(self):
        return self.W

class Net4(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter((torch.rand(6) - 0.5) * 2.)
    def forward(self,x):
        w1 = self.W[0:4].view((1, 4))
        x = F.linear(x,w1)
        w2 = self.W[4:6].view((2,1))
        x = F.linear(x,w2)
        return x
    def get_W(self):
        return self.W
#%% IRIS

repeats = 10
n_epochs = 100
n_data = 100
X,Y =  get_iris()
losses = np.zeros((repeats, n_epochs))
losses2 = np.zeros((repeats, n_epochs))
losses3 = np.zeros((repeats, n_epochs))
for r in tqdm(range(repeats)):
    net = Net4()
    optim = torch.optim.Adam(params = net.parameters(),lr=0.1)
    for e in range(100):
        optim.zero_grad()
        pred = net(X)
        loss = F.cross_entropy(pred,Y)
        loss.backward()
        optim.step()
        losses[r,e] = loss.detach().item()
    net2 = Net4112()
    optim2 = torch.optim.Adam(params=net2.parameters(), lr=0.1)
    for e in range(100):
        optim2.zero_grad()
        pred = net2(X)
        loss = F.cross_entropy(pred, Y)
        loss.backward()
        optim2.step()
        losses2[r, e] = loss.detach().item()
    net3 = Net42()
    optim3 = torch.optim.Adam(params=net3.parameters(), lr=0.1)
    for e in range(100):
        optim3.zero_grad()
        pred = net3(X)
        loss = F.cross_entropy(pred, Y)
        loss.backward()
        optim3.step()
        losses3[r, e] = loss.detach().item()
#%%


plt.plot(range(100),losses.mean(axis=0))
plt.fill_between(range(100),losses.mean(axis=0)-losses.std(axis=0),losses.mean(axis=0)+losses.std(axis=0),alpha=0.2)
plt.plot(range(100),losses2.mean(axis=0))
plt.fill_between(range(100),losses2.mean(axis=0)-losses2.std(axis=0),losses2.mean(axis=0)+losses2.std(axis=0),alpha=0.2)
plt.plot(range(100),losses3.mean(axis=0))
plt.fill_between(range(100),losses3.mean(axis=0)-losses3.std(axis=0),losses3.mean(axis=0)+losses3.std(axis=0),alpha=0.2)

plt.show()

#%%
n_iters = 50
X,Y = get_iris()
n_data = 100
Fishers = np.zeros((n_iters,DIM,DIM))
efs={100: []}
for r in range(10):
    for j in range(n_iters):
        net = Net42()
        Fish = np.zeros((n_data,DIM,DIM))
        for i,(x,y) in enumerate(zip(X,Y)):
            pred = net(x.view((1,4)))
            loss = F.cross_entropy(pred,y.view((1)))
            Fish[i] = fisher(loss,net.get_W())
        Fishers[j] = Fish.mean(axis=0)
    ef = effective_dimension(normalise_fishers(Fishers), n_data)
    efs[n_data].append(ef)
    print(f"{n_data}: {ef}")

#%%
repeats = 10
n_epochs = 100
n_data = 100
X,Y = initialise_data(n_data)
X_val, Y_val = initialise_data(n_data)
losses = np.zeros((repeats, n_epochs, 2))
for r in tqdm(range(repeats)):
    net = Net4442()
    optim = torch.optim.Adam(params = net.parameters(),lr=0.1)
    for e in range(100):
        optim.zero_grad()
        pred = net(X)
        loss = F.cross_entropy(pred,Y)
        loss.backward()
        optim.step()
        losses[r,e,0] = loss.detach().item()
        with torch.no_grad():
            val_pred = net(X_val)
            val_loss = F.cross_entropy(val_pred,Y_val)
            losses[r,e,1] = val_loss.item()

#%%
plt.plot(range(100),losses.mean(axis=0)[:,0])
plt.fill_between(range(100),losses.mean(axis=0)[:,0]-losses.std(axis=0)[:,0],losses.mean(axis=0)[:,0]+losses.std(axis=0)[:,0],alpha=0.2)
plt.plot(range(100),losses.mean(axis=0)[:,1])
plt.fill_between(range(100),losses.mean(axis=0)[:,1]-losses.std(axis=0)[:,1],losses.mean(axis=0)[:,1]+losses.std(axis=0)[:,1],alpha=0.2)


plt.show()

#%%
DIM=40
efs={}
for n_data in [100]*5:
    n_iters = 50
    X,Y = initialise_data(n_data)

    Fishers = np.zeros((n_iters,DIM,DIM))
    for j in range(n_iters):
        net = Net4442()
        Fish = np.zeros((n_data,DIM,DIM))
        for i,(x,y) in enumerate(zip(X,Y)):
            pred = net(x.view((1,4)))
            loss = F.cross_entropy(pred,y.view((1)))
            Fish[i] = fisher(loss,net.get_W())
        Fishers[j] = Fish.mean(axis=0)
    if n_data in efs.keys() and type(efs[n_data]) == list: efs[n_data].append(effective_dimension(normalise_fishers(Fishers), n_data))
    elif n_data in efs.keys(): efs[n_data] = [efs[n_data]]+[effective_dimension(normalise_fishers(Fishers), n_data)]
    else: efs[n_data] = effective_dimension(normalise_fishers(Fishers), n_data)
    print(f"{n_data}: {efs[n_data]}")

