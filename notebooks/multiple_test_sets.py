import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import matplotlib
font = {'family': 'serif', 'size': 18}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

test = np.load(open("../data/qnn_loss_pulsardata (1).npy","rb"))


n_iteration = 10
num_epochs = 100
depths = [0, ]
variations = ['RYRX']

qnn_losses = { k: {dd : {"Train" : np.zeros(n_iterations,num_epochs), } for dd in depths} for k in variations}
