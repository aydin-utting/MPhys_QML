# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:53:06 2021

@author: Aydin Utting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import torch
from scipy.special import logsumexp
from pennylane import numpy as  np
# PennyLane for QNN
import pennylane as qml

# tqdm for progress bars
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

# Set global dimension
DIM = 16
# Random seed for reproducability
seed = 13


def get_data(N):
    # load HTRU_2 dataset
    dataset = pd.read_csv('../data/HTRU_2.csv', names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'Species'])

    for k in dataset[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]:
        dataset[k] = minmax_scale(dataset[k], feature_range=(-1., 1.))

    data0 = dataset[dataset[dataset.columns[8]] == 0]
    data0 = data0.sample(n=N, random_state=seed)
    X0 = data0[data0.columns[0:8]].values
    Y0 = data0[data0.columns[8]].values

    data1 = dataset[dataset[dataset.columns[8]] == 1]
    data1 = data1.sample(n=N, random_state=seed)
    X1 = data1[data1.columns[0:8]].values
    Y1 = data1[data1.columns[8]].values

    X = np.append(X0, X1, axis=0)
    Y = np.append(Y0, Y1, axis=0)

    train_X, validation_X, train_Y, validation_Y = train_test_split(X, Y, test_size=0.5, random_state=seed)
    train_X = np.array(train_X, requires_grad=False)
    train_Y = np.array(train_Y, requires_grad=False)
    data = list(zip(train_X, train_Y))
    return data


dev = qml.device("default.qubit.autograd", wires=9)


@qml.qnode(dev, diff_method='backprop')
def quantum_neural_network(x, w, depth, variation):
    # encoding circuit============================================================

    # Hadamards

    # Accounting for depth=0
    if (depth == 0):
        for i in range(8):
            qml.Hadamard(wires=i)
        for i in range(8):
            qml.RZ(x[i], wires=i)

    # Multiple encoding layers:
    for k in range(depth):

        for i in range(8):
            qml.Hadamard(wires=i)

        # RZ gates
        for i in range(8):
            qml.RZ(x[i], wires=i)
        # RZZ gates
        for i in range(8):
            for j in range(i):
                qml.CNOT(wires=[j, i])
                qml.RZ(((np.pi - x[i]) * (np.pi - x[j])), wires=[i])
                qml.CNOT(wires=[j, i])

    # variational circuit =======================================================
    for i in range(8):
        qml.RY(w[0][i], wires=i)

    qml.broadcast(qml.CNOT, wires=range(8), pattern="all_to_all")

    for i in range(8):
        if variation == "RYRY":
            qml.RY(w[1][i], wires=i)
        elif variation == "RYRX":
            qml.RX(w[1][i], wires=i)

    dev.shots = 10000

    for i in range(8):
        qml.CNOT(wires=[i, 8])

    return qml.expval(qml.PauliZ(wires=8))


def get_parity_prediction(x, w, depth, variation):
    np_measurements = (quantum_neural_network(x, w, depth, variation) + 1.) / 2.

    return np.array([np_measurements, 1. - np_measurements])


def single_loss(w, x, y, depth, variation):
    prediction = get_parity_prediction(x, w, depth, variation)
    return rel_ent(prediction, y)


def rel_ent(pred, y):
    return -np.log(pred)[int(y)]


def average_loss(w, data, depth: int, variation: str):
    c = 0
    Fisher = np.zeros((DIM, DIM))
    for i, (x, y) in enumerate(data):
        single_cost = single_loss(w, x, y, depth, variation)
        c += single_cost
        grad_fn = qml.grad(single_loss, argnum=0)
        gradient = grad_fn(w, x, y, depth, variation).flatten()
        Fisher += np.outer(gradient, gradient)
    return c / len(data), Fisher / len(data)


def get_all_fishers(n_iter: int, depth: int, variation: str, data):
    all_fishers = np.zeros((n_iter, DIM, DIM))
    pbar = tqdm(range(n_iter), desc=f"{variation} Depth: {depth} ", leave=False)
    for i in pbar:
        w = np.array(np.split(np.random.uniform(size=(DIM,), low=-1.0, high=1.0), 2), requires_grad=True)
        Fisher = average_loss(w, data, depth, variation)[1]
        all_fishers[i] = Fisher
    return all_fishers


def normalise_fishers(Fishers):
    TrF_integral = Fishers.trace(axis1=1, axis2=2).mean()
    return (DIM / TrF_integral) * Fishers


def effective_dimension_amira(normed_fishers, n_data: int):
    print(f"normedf.shape: {normed_fishers.shape}")
    id = torch.eye(DIM)
    kappa = (n_data) / (2 * np.pi * np.log(n_data))
    FHat = id + kappa * normed_fishers
    det = np.linalg.slogdet(id + kappa * FHat)[1]
    print(f"det.shape: {det}")
    numerator = logsumexp(det / 2.) - np.log(len(normed_fishers))
    eff_dim = 2 * numerator / np.log(kappa)
    return eff_dim


def effective_dimension(normed_fishers, n_data: int):
    id = torch.eye(DIM)
    kappa = n_data / (2 * np.pi * np.log(n_data))
    integral = np.average(np.sqrt(np.linalg.det(id + kappa * normed_fishers)))
    numerator = np.log(integral)
    return 2 * numerator / np.log(kappa)


def get_entropy(depth: int, variation: str, n_iter: int, bins: int):
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


def get_effective_dimension(depth: int, variation: str, n_iter: int, n_data: int, data):
    all_fishers = get_all_fishers(n_iter, depth, variation, data)
    normalised_fishers = normalise_fishers(all_fishers)
    d_eff = effective_dimension(normalised_fishers, n_data)
    return d_eff


if __name__ == '__main__':
    n_data=300
    data = get_data(n_data)
    repeats = 1
    depths = [0, 1, 2, 3, 4, 5]
    n_iters = 10
    variations = ["RYRY","RYRX"]
    depth_effective_dimensions = {k: {dd: [0.0 for _ in range(repeats)] for dd in depths} for k in variations}
    for r in range(repeats):
        for depth in depths:
            for vv, variation in enumerate(variations):
                depth_effective_dimensions[variation][depth][r] = get_effective_dimension(depth, variation, n_iters,n_data, data)
                pk.dump(depth_effective_dimensions, open("../data/depth_effective_dimensions_pulsars_300.data", "wb"))
                print(f"{variation}{depth} : {depth_effective_dimensions[variation][depth][r]}")
