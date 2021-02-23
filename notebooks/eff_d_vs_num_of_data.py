import pennylane as qml
from pennylane import numpy as np
import copy

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd

# Data initialisation
def create_data(n_samples,var):
    variance = var
    X0 =np.array([[np.random.normal(loc=-1, scale=variance),
                    np.random.normal(loc=1, scale=variance),
                    np.random.normal(loc=-1, scale=variance),
                    np.random.normal(loc=1, scale=variance)] for i in range(int(n_samples/2))],requires_grad=False)


    X1 = np.array([[np.random.normal(loc=1, scale=variance),
                    np.random.normal(loc=-1, scale=variance),
                    np.random.normal(loc=1, scale=variance),
                    np.random.normal(loc=-1, scale=variance)] for i in range(int(n_samples/2))],requires_grad=False)

    X = np.append(X0, X1,0)

    Y = np.append(np.array([0 for i in range(int(n_samples/2))],requires_grad=False),np.array([1 for i in range(int(n_samples/2))],requires_grad=False))

    data = list(zip(X, Y))
    return data


dev = qml.device("default.qubit", wires=4)
@qml.qnode(dev, diff_method='backprop')
def quantum_neural_network(x, w):
    qml.templates.IQPEmbedding(x, wires=[0,1,2,3], n_repeats=2, pattern=None)
    for j in range(10):
      for i in range(4):
        qml.RY(w[j][i],wires=i)
      qml.broadcast(qml.CNOT, wires=[0,1,2,3], pattern="all_to_all", parameters=None, kwargs=None)
    dev.shots = 1000
    sample_measurements = [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]

    return sample_measurements


dev = qml.device("default.qubit", wires=4)
@qml.qnode(dev, diff_method='backprop')
def easy_quantum(x, w):
    qml.templates.IQPEmbedding(x, wires=[0,1,2,3], n_repeats=2, pattern=None)

    for j in range(10):
      for i in range(4):
        qml.RY(w[j][i],wires=[i])
    dev.shots = 1000
    sample_measurements = [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]

    return sample_measurements


def parameter_shift_term(qnode, model, params, x, y, i, j, s):
    shifted = params.copy()

    shifted[i, j] += s
    forward = qnode(model, shifted, x, y)  # forward evaluation

    shifted[i, j] -= 2 * s
    backward = qnode(model, shifted, x, y)  # backward evaluation

    return 0.5 * (forward - backward) / np.sin(s)


def parameter_shift(qnode, model, params, x, y, s):
    gradients = np.zeros_like((params))

    for i in range(len(gradients)):
        for j in range(len(gradients[0])):
            gradients[i, j] = parameter_shift_term(qnode, model, params, x, y, i, j, s)

    return gradients


def average_loss(data,w, shift, model):
    c = 0.
    n_samples = len(data)
    Fisher = np.zeros((40, 40))
    for i, (x, y) in enumerate(tqdm(data)):
        grad_func = qml.grad(single_loss,argnum=1)
        gradient = grad_func(model,w,x,y)
        c += single_loss(model, w, x, y)
        # print(quantum_neural_network.draw())
        #single_grad = parameter_shift(single_loss, model, w, x, y, shift)
        Fisher += np.outer(gradient.flatten(), gradient.flatten())
    return c / n_samples, Fisher / n_samples




def get_parity_prediction(model, x,w):

    #choice of models
    if model=="QNN": np_measurements = quantum_neural_network(x,w)
    else: np_measurements= easy_quantum(x,w)

    probability_vector=(np_measurements+1.)/2.
    p_0 = prob_parity_even(probability_vector)
    p_1 = 1.-p_0
    #parity_vector = (np.abs(np.sum(np_measurements,axis=0))/2)%2.0
    #unique, counts = np.unique(parity_vector, return_counts=True)
    #return counts/sum(counts)
    return np.array([p_0,p_1])


def prob_parity_even(m):
    p = 0.0
    #all down
    p+=(1.-m).prod()
    #2 down 2 up
    for i in range(len(m)):
        for j in range(0,i):
            a = np.zeros_like(m)
            a[i]=1.
            a[j]=1.
            p+=(a-m).prod()
            """copy_m = m
            copy_m[i] = 1.-copy_m[i]
            copy_m[j] = 1.-copy_m[j]
            p+=copy_m.prod()"""

    #all up
    p+=m.prod()
    return p

def prob_parity_odd(m):
    p=0.0
    #3 up 1 down
    for i in range(len(m)):
        copy_m = m
        copy_m[i] = 1. - copy_m[i]
        p+=copy_m.prod()
    #3 down 1 up
    for i in range(len(m)):
        copy_m = m
        copy_m = 1.-copy_m
        copy_m[i] = 1.-copy_m[i]
        p += copy_m.prod()
    return p

def single_loss(model, w,x,y):
    prediction = get_parity_prediction(model, x,w)
    return rel_ent(prediction, y)

def rel_ent(pred,y):
    return -1.*np.log(pred)[y]

# Function to normalise Fisher informations
def normalise_fishers(Fishers, thetas, d, V):
    num_samples = len(Fishers)

    TrF_integral = (1 / num_samples) * np.sum(np([np.trace(F) for F in Fishers]))
    return [((d * V) / TrF_integral) * F for F in Fishers]

# Function to calculate effective dimension
def effective_dimension(normed_fishers, n):
    d = 40
    V_theta = 1
    gamma = 1
    id = np.eye(d)
    kappa = np.array([(gamma * n) / (2 * np.pi * np.log(n))])
    integral = np.array([0.0])
    for F in normed_fishers:
        integral += np.sqrt(np.linalg.det(id + kappa * F))
    integral_over_volume = integral / (V_theta*len(normed_fishers))
    numerator = np.log(integral_over_volume)
    return 2 * numerator / np.log(kappa)


def sample_theta_for_fisher(n_iter, data,shift,model, plots=False):
    EV = []
    FR = []
    Rank = []
    all_w = []
    all_fishers = []
    for i in tqdm(range(n_iter)):
        w = np.array(np.split(np.random.uniform(size=(40,),low=-1.0,high=1.0),10),requires_grad=True)
        total_loss, Fisher = average_loss(data, w,shift,model)
        val, v = np.linalg.eig(Fisher)
        EV.append(np.real(val))
        all_w.append(w)
        all_fishers.append(Fisher)
        Rank.append(np.linalg.matrix_rank(Fisher).item())
        Fw = np.matmul(Fisher, w.flatten())
        wFw = np.dot(w.flatten(), Fw)
        FR.append(wFw)
    if plots:
        print("Fisher")
        x, bins, p = plt.hist(EV, bins=None, range=None)
        for item in p:
            item.set_height(item.get_height() / (40 * n_iter))
        plt.ylim(0, 1)
        plt.title("Sigmoid")
        plt.show()

        plt.hist(Rank)
        plt.title('matrix ranks for all Fisher')
        plt.show()
        plt.hist(FR)
        plt.title('Fisher-Rao norm for all Fisher')
        plt.show()

    return EV, FR, Rank, all_w, all_fishers




runs = 1
efs = pd.DataFrame(index=list(range(runs)), columns=list(range(10, 110, 10)))
data = create_data(10,1.)
for r in range(runs):
    for n_iters in range(100, 110, 10):
        EV, FR, Rank, all_w, all_fishers = sample_theta_for_fisher(n_iters, data,np.pi/2,"QNN", plots=False)
        nF = normalise_fishers(all_fishers, all_w, 40, 1)
        ef = effective_dimension(nF, len(data))
        print('D_EFF = ', ef)
        efs.loc[r, n_iters] = ef.item()


