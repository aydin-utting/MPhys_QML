# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:53:06 2021

@author: Aydin Utting
"""
from scipy.special import logsumexp

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle as pk
# Torch for Classical NN
import torch
import torch.nn as nn
import torch.nn.functional as F

# PennyLane for QNN
import pennylane as qml
from pennylane.optimize import AdamOptimizer

# Qiskit for backend
from qiskit import IBMQ

#tqdm for progress bars
from tqdm.auto import tqdm
from tqdm._utils import _term_move_up
#Set global dimension
DIM = 8

# load IRIS dataset
dataset = pd.read_csv('../data/iris.data',names=['a','b','c','d','Species'])

# transform species to numerics
dataset.loc[dataset.Species == 'Iris-setosa', 'Species'] = 0
dataset.loc[dataset.Species == 'Iris-versicolor', 'Species'] = 1
dataset.loc[dataset.Species == 'Iris-virginica', 'Species'] = 2
dataset = dataset.query('Species==0 or Species==1')

# normalise
x = dataset[dataset.columns[0:4]].values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaled = min_max_scaler.fit_transform(x)

#df = pd.DataFrame(x_scaled, columns=dataset.columns[0:4])


#train_X, train_Y = df.astype(float).values, dataset.Species.astype(int).values
train_X, train_Y = x_scaled, dataset.Species.astype(int).values

num_epochs = 100
loss_curve_x = np.array(range(num_epochs))


##----------------------------------------CLASSICAL NN ----------------------------------------------
def classical_iris_train(train_X, train_Y):
    from torch.autograd import Variable
    class Net(nn.Module):
        # define nn
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(4, 2, bias=False)
            nn.init.uniform_(self.fc1.weight, -1., 1)

        def forward(self, X):
            X = self.fc1(X)
            return X

    # wrap up with Variable in pytorch
    torch_train_X = Variable(torch.Tensor(train_X).float())

    torch_train_y = Variable(torch.Tensor(np.float64(train_Y)).long())

    net = Net()

    criterion = nn.CrossEntropyLoss()  # cross entropy loss
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    cnn_loss_curve_y = np.array([])
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = net(torch_train_X)
        loss = criterion(out, train_Y)
        loss.backward()
        optimizer.step()
        cnn_loss_curve_y = np.append(cnn_loss_curve_y, loss.item())
        # if epoch % 100 == 0:
        # print('number of epoch '+  str(epoch) + ' loss '+ str(loss.data.item()))

    return cnn_loss_curve_y


##---------------------------------------------------------------------------------------------------


# ----------------------------------------QUANTUM NN-------------------------------------------------

from pennylane import numpy as  np

train_X = np.array(train_X, requires_grad=False)
train_Y = np.array(train_Y, requires_grad=False)

data = list(zip(train_X, train_Y))

choice_of_device = 'ibmq_simulator'  # choice from ibmq, ibmq_simulator, pennylane

'''if "ibm" in choice_of_device:
    ibmq_token = "633f0df51089fa6fb75dc3073bbbb551452172a5f8165e03cd99ff42eb9c322d50b891dbb87aa1c9ea05877d500d09af4331d7ac05f1e160a2274e8f00523ba7"
    IBMQ.enable_account(ibmq_token, overwrite=True)
    my_provider = IBMQ.get_provider()
    my_backend = my_provider.get_backend('ibmq_qasm_simulator')
    if (choice_of_device == 'ibmq_simulator'):
        dev = qml.device("qiskit.ibmq", wires=5, backend=my_backend, ibmqx_token=ibmq_token, provider=my_provider)
    else:
        provider = IBMQ.get_provider(hub='ibm-q')
        dev = qml.device("qiskit.ibmq", wires=5, backend='ibmqx2', ibmqx_token=ibmq_token)

else:
'''
dev = qml.device("default.qubit.autograd", wires=5)


@qml.qnode(dev, diff_method='backprop', wires=5)
def quantum_neural_network(x, w, depth, variation):
    # #Hadamards

    # encoding circuit============================================================

    # Hadamards
    for i in range(4):
        qml.Hadamard(wires=i)

    # Accounting for depth=0
    if (depth == 0):
        for i in range(4):
            qml.RZ(x[i], wires=i)

    # Multiple encoding layers:
    for k in range(depth):
        # RZ gates
        for i in range(4):
            qml.RZ(x[i], wires=i)
        # RZZ gates
        for i in range(4):
            for j in range(i):
                qml.CNOT(wires=[j, i])
                qml.RZ(((x[i]) * (x[j])), wires=[i])
                qml.CNOT(wires=[j, i])

    # variational circuit ==========================================================
    for j in range(4):
        qml.RY(w[0][j], wires=j)

    qml.broadcast(qml.CNOT, wires=[0, 1, 2, 3], pattern="all_to_all")

    for i in range(4):
        if variation == "RYRX":
            qml.RX(w[1][i], wires=i)
        elif variation == "RYRY":
            qml.RY(w[1][i], wires=i)

    dev.shots = 1000

    for i in range(4):
        qml.CNOT(wires=[i, 4])

    return qml.expval(qml.PauliZ(wires=4))


def get_parity_prediction(x, w, depth, variation):
    np_measurements = (quantum_neural_network(x, w, depth, variation) + 1.) / 2.

    return np.array([np_measurements, 1. - np_measurements])


def single_loss(w, x, y, depth, variation):
    prediction = get_parity_prediction(x, w, depth, variation)
    return rel_ent(prediction, y)


def rel_ent(pred, y):
    return -np.log(pred)[int(y)]


def average_loss(w, data, depth, variation,iter=None):
    c = 0
    Fisher = np.zeros((DIM, DIM))
    for i, (x, y) in enumerate(data):
        single_cost = single_loss(w, x, y, depth, variation)
        c += single_cost
        grad_fn = qml.grad(single_loss, argnum=0)
        gradient = grad_fn(w, x, y, depth, variation).flatten()
        Fisher += np.outer(gradient, gradient)
    tqdm._instances.clear()
    return c / len(data), Fisher / len(data)


def get_all_fishers(n_iter, depth=2, variation="RYRY"):
    all_fishers = np.zeros((n_iter, DIM, DIM))
    pbar = tqdm(range(n_iter), desc=f"{variation} Depth: {depth} ", leave=False)
    for i in pbar:
        w = np.array(np.split(np.random.uniform(size=(DIM,), low=-1., high=1.), 2), requires_grad=True)
        Fisher = average_loss(w, data, depth, variation,i)[1]
        all_fishers[i] = Fisher
    return all_fishers


def normalise_fishers(Fishers):
    num_samples = len(Fishers)
    TrF_integral = (1 / num_samples) * np.sum([np.trace(F) for F in Fishers])
    return [((DIM) / TrF_integral) * F for F in Fishers]


def effective_dimension_good(normed_fishers, n_data):
    d = DIM
    V_theta = 1.
    gamma = 1.
    id = torch.eye(d)
    kappa = (gamma * n_data) / (2 * np.pi * np.log(n_data))
    integral = 0.0
    for Fish in normed_fishers:
        integral += np.sqrt(np.linalg.det(id + kappa * Fish))

    integral_over_volume = integral / (V_theta * len(normed_fishers))
    numerator = np.log(integral_over_volume)
    return 2 * numerator / np.log(kappa)

def effective_dimension(normed_fishers, n_data : int):
    normed_fishers = np.array(normed_fishers)
    id = torch.eye(DIM)
    kappa = (n_data) / (2 * np.pi * np.log(n_data))
    FHat = id + kappa * normed_fishers
    det = np.linalg.slogdet(FHat)[1]
    numerator = logsumexp(det/2.) - np.log(len(normed_fishers))
    eff_dim = 2 * numerator / np.log(kappa)
    return eff_dim


def get_entropy(depth=2, variation="RYRY", n_iter=5, bins=10):
    all_fishers = get_all_fishers(n_iter, depth, variation)
    normalised_fishers = normalise_fishers(all_fishers)
    EVs = np.array([])
    for F in normalised_fishers:
        val, v = np.linalg.eig(F)
        EVs = np.append(np.real(EVs), val)
    x, bins, p = plt.hist(EVs, bins=bins)
    entropy = 0
    x = x / len(EVs)
    for i in x:
        if (i != 0):
            entropy -= i * np.log(i)
    return entropy

def get_effective_dimension(depth=2, variation="RYRY", n_iter=5, n_data=1000):
    all_fishers = get_all_fishers(n_iter, depth, variation)
    normalised_fishers = normalise_fishers(all_fishers)
    d_eff = effective_dimension(normalised_fishers,n_data)
    return d_eff

repeats = 1
depths = [5]
n_iters = 5
n_data = len(train_X)
variations = ["RYRY"]
depth_effective_dimensions = { k: {dd : [0.0 for _ in range(repeats)] for dd in depths} for k in variations}
for r in range(repeats):
    for depth in depths:
        for vv,variation in enumerate(variations):
            depth_effective_dimensions[variation][depth][r] = get_effective_dimension(depth,variation,n_iters,n_data)
            pk.dump(depth_effective_dimensions,open("../data/depth_effective_dimensions_test.data","wb"))
            print(f"{variation} Depth: {depth} d_eff = {depth_effective_dimensions[variation][depth][r]}")

#NOTE NEED TO CHANGE quantum_neural_network