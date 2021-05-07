# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:53:06 2021

@author: Aydin Utting
"""

import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk
import torch
from scipy.special import logsumexp
from pennylane import numpy as  np
# PennyLane for QNN
import pennylane as qml

#tqdm for progress bars
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import minmax_scale

#Set global dimensions
DIM = 40
S_IN = 4
L = DIM // S_IN
#Random seed for reproducability
seed = 13


def initialise_data(n_samples):
    variance = 1
    X0 = np.array([[np.random.normal(loc=-1, scale=variance),
                    np.random.normal(loc=1, scale=variance),
                    np.random.normal(loc=-1, scale=variance),
                    np.random.normal(loc=1, scale=variance)] for i in range(int(n_samples / 2))], requires_grad=False)

    X1 = np.array([[np.random.normal(loc=1, scale=variance),
                    np.random.normal(loc=-1, scale=variance),
                    np.random.normal(loc=1, scale=variance),
                    np.random.normal(loc=-1, scale=variance)] for i in range(int(n_samples / 2))], requires_grad=False)

    X = np.append(X0, X1, 0)

    Y = np.append(np.array([0 for i in range(int(n_samples / 2))], requires_grad=False),
                  np.array([1 for i in range(int(n_samples / 2))], requires_grad=False))

    data = list(zip(X, Y))

    return data


# ----------------------------------------QUANTUM NN-------------------------------------------------
data = initialise_data(100)



dev = qml.device("default.qubit.autograd", wires=S_IN + 1)

@qml.qnode(dev, diff_method='backprop')
def quantum_neural_network(x, w, depth ,variation):
    # encoding circuit============================================================

    # Hadamards
    for i in range(S_IN):
        qml.Hadamard(wires=i)

    # Accounting for depth=0
    if (depth == 0):
        for i in range(S_IN):
            qml.RZ(x[i], wires=i)

    # Multiple encoding layers:
    for k in range(depth):
        # RZ gates
        for i in range(S_IN):
            qml.RZ(x[i], wires=i)
        # RZZ gates
        for i in range(S_IN):
            for j in range(i):
                qml.CNOT(wires=[j, i])
                qml.RZ(((x[i]) * (x[j])), wires=[i])
                qml.CNOT(wires=[j, i])

    # variational circuit =======================================================


    for l in range(L-1):

        for i in range(S_IN):
            qml.RY(w[l][i], wires=i)

        qml.broadcast(qml.CNOT, wires=range(S_IN), pattern="all_to_all")

    for i in range(4):
        qml.RY(w[L-1][i], wires=i)


    dev.shots = 10000

    for i in range(S_IN):
        qml.CNOT(wires=[i, S_IN])

    return qml.expval(qml.PauliZ(wires=S_IN))

def get_parity_prediction(x, w, depth, variation):
    np_measurements = (quantum_neural_network(x, w, depth, variation) + 1.) / 2.

    return np.array([np_measurements, 1. - np_measurements])


def single_loss(w, x, y, depth, variation):
    prediction = get_parity_prediction(x, w, depth, variation)
    return rel_ent(prediction, y)


def rel_ent(pred, y):
    return -np.log(pred)[int(y)]


def average_loss(w, data, depth : int, variation : str):
    c = 0
    Fisher = np.zeros((DIM, DIM))
    for i, (x, y) in enumerate(data):
        single_cost = single_loss(w, x, y, depth, variation)
        c += single_cost
        grad_fn = qml.grad(single_loss, argnum=0)
        gradient = grad_fn(w, x, y, depth, variation).flatten()
        Fisher += np.outer(gradient, gradient)
    return c / len(data), Fisher / len(data)


def get_all_fishers(n_iter : int, depth : int, variation : str):
    all_fishers = np.zeros((n_iter, DIM, DIM))
    pbar = tqdm(range(n_iter), desc=f"{variation} Depth: {depth} ", leave=False)
    for i in pbar:
        w = np.array(np.split(np.random.uniform(size=(DIM,), low=-1.0, high=1.0), L), requires_grad=True)
        Fisher = average_loss(w, data, depth, variation)[1]
        all_fishers[i] = Fisher
    return all_fishers


def normalise_fishers(Fishers):

    TrF_integral = Fishers.trace(axis1=1,axis2=2).mean()
    return (DIM/TrF_integral) * Fishers

def effective_dimension_amira(normed_fishers, n_data : int):
    normed_fishers = np.array(normed_fishers)
    print(f"normedf.shape: {normed_fishers.shape}")
    id = torch.eye(DIM)
    kappa = (n_data) / (2 * np.pi * np.log(n_data))
    FHat = id + kappa * normed_fishers
    det = np.linalg.slogdet(id + kappa * FHat)[1]
    print(f"det.shape: {det}")
    numerator = logsumexp(det/2.) - np.log(len(normed_fishers))
    eff_dim = 2 * numerator / np.log(kappa)
    return eff_dim

def effective_dimension(normed_fishers, n_data : int):
    normed_fishers = np.array(normed_fishers)
    id = torch.eye(DIM)
    kappa = n_data / (2 * np.pi * np.log(n_data))
    integral = np.average(np.sqrt(np.linalg.det(id + kappa * normed_fishers)))
    numerator = np.log(integral)
    return 2 * numerator / np.log(kappa)

def get_entropy(depth : int, variation : str, n_iter : int, bins : int):
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

def get_effective_dimension(depth : int, variation : str, n_iter : int, n_data : int):
    all_fishers = get_all_fishers(n_iter, depth, variation)
    normalised_fishers = normalise_fishers(all_fishers)
    d_eff = effective_dimension(normalised_fishers,n_data)
    return d_eff

if __name__ == '__main__':

    repeats = 1
    depths = [5]
    n_iters = 5
    n_data = len(data)
    variations = ["RYRY"]
    depth_effective_dimensions = { k: {dd : [0.0 for _ in range(repeats)] for dd in depths} for k in variations}
    for r in range(repeats):
        for depth in depths:
            for vv,variation in enumerate(variations):
                depth_effective_dimensions[variation][depth][r] = get_effective_dimension(depth,variation,n_iters,n_data)
                pk.dump(depth_effective_dimensions,open("../data/depth_effective_dimensions_temp.data","wb"))
                print(f"{depth_effective_dimensions[variation][depth][r]}")