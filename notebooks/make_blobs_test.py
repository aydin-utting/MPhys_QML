#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from tqdm.auto import tqdm
import random

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


def normalise_fishers(Fishers, d, V):
    num_samples = len(Fishers)

    TrF_integral = (1 / num_samples) * torch.sum(torch.tensor([torch.trace(F) for F in Fishers]))
    return [((d * V) / TrF_integral) * F for F in Fishers]


# Function to calculate effective dimension
def effective_dimension(normed_fishers, n):
    d = normed_fishers[0].size()[0]
    V_theta = 1.
    gamma = 1.
    id = torch.eye(d)
    kappa = torch.tensor([(gamma * n) / (2 * np.pi * np.log(n))])
    integral = torch.tensor([0.0])
    for Fish in normed_fishers:
        integral += torch.sqrt(torch.det(id + kappa * Fish))

    integral_over_volume = integral / (V_theta * len(normed_fishers))
    numerator = torch.log(integral_over_volume)
    return 2 * numerator / torch.log(kappa)

colours = [
    'r','b','g'
]
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

X,Y = make_blobs(1000,6,centers=2,center_box=(-5.,5.),random_state=42)
X,Y = torch.tensor(X).double(),torch.tensor(Y).int()

class Net(nn.Module):
    def __init__(self,W):
        super().__init__()
        self.W = nn.Parameter(W)
    def forward(self,x):
        w1 = self.W[0:660].view((110,6))
        x = F.sigmoid(F.linear(x,w1))
        w2 = self.W[660:880].view((2,110))
        x = F.linear(x,w2)
        return x


# TRAIN
class Confuser():
    def __init__(self):
        pass

    def __call__(self,y, num_classes,fraction_confused):
        T = y.clone()
        indx = random.sample(range(len(y)),int(len(y)*fraction_confused))
        for i in indx:
            T[i] = torch.randint(0,num_classes,(1,))
        return T

conf_percentages = [0.1,0.2,0.3,0.4,0.5]
labelsets = {0.0 : Y}
conf = Confuser()
for p in conf_percentages:
    labelsets[p] = conf(Y,2,p)
#%%
deffs = {}
for k in labelsets:
    runs = 100
    models = {}
    for r in range(runs):
        W = (torch.rand(880)-0.5)*2.
        model = Net(W).double()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
        epochs = 1000
        loss = torch.tensor([10.])
        pbar = tqdm(range(epochs),desc=str(k))
        train_acc = 1e10
        for epoch in pbar:
            optimizer.zero_grad()
            pred = model(X)
            loss_criterion = nn.CrossEntropyLoss()
            loss = loss_criterion(pred,labelsets[k].long())
            loss.backward()
            optimizer.step()
            prev_train_acc = train_acc
            train_acc = (sum(F.softmax(pred, dim=1).argmax(dim=1) == labelsets[k]) / len(labelsets[k])).item()
            pbar.set_postfix( {"Training acc" : train_acc, "Loss" : loss.item()} )
            if train_acc == 1.: break
            elif abs(train_acc-prev_train_acc) ==0. and train_acc>0.6: break
        models[r] = model

    all_fishers = []
    for m in tqdm(models):
        pred = models[m](X)
        loss_criterion = nn.CrossEntropyLoss()
        loss = loss_criterion(pred,labelsets[k].long())
        J = jacobian(loss,models[m].W)
        Fisher = torch.outer(J,J.T)
        all_fishers.append(Fisher)

    nF = normalise_fishers(all_fishers,880,1.)
    d_eff = effective_dimension(nF, X.size()[0])
    print(d_eff)
    deffs[k] = d_eff

#%%
# untrained calculation of effective dimension
cbox = range(1,11)
deffs = {}
models = {r : Net((torch.rand(880)-0.5)*2.).double() for r in range(100)}
for cb in cbox:
    X,Y = make_blobs(1000,6,centers=2,center_box=(-1.*cb,cb),random_state=42)
    X,Y = torch.tensor(X).double(),torch.tensor(Y).int()
    all_fishers = []

    for m in tqdm(models):
        pred = models[m](X)
        loss_criterion = nn.CrossEntropyLoss()
        loss = loss_criterion(pred,Y.long())
        J = jacobian(loss,models[m].W)
        Fisher = torch.outer(J,J.T)
        all_fishers.append(Fisher)

    nF = normalise_fishers(all_fishers,880,1.)
    d_eff = effective_dimension(nF, X.size()[0])
    print(d_eff)
    deffs[cb] = d_eff