import pennylane as qml
from pennylane import numpy as np

def initialise_data():
    n_samples = 100
    variance = (np.pi / 4) ** 2
    X0 = np.array([[np.random.normal(loc=-np.pi / 4, scale=variance),
                    np.random.normal(loc=np.pi / 4, scale=variance),
                    np.random.normal(loc=-np.pi / 4, scale=variance),
                    np.random.normal(loc=np.pi / 4, scale=variance)] for i in range(int(n_samples / 2))],
                  requires_grad=False)

    X1 = np.array([[np.random.normal(loc=np.pi / 4, scale=variance),
                    np.random.normal(loc=-np.pi / 4, scale=variance),
                    np.random.normal(loc=np.pi / 4, scale=variance),
                    np.random.normal(loc=-np.pi / 4, scale=variance)] for i in range(int(n_samples / 2))],
                  requires_grad=False)

    X = np.append(X0, X1, 0)

    Y = np.append(np.array([0 for i in range(int(n_samples / 2))], requires_grad=False),
                  np.array([1 for i in range(int(n_samples / 2))], requires_grad=False))

    data = list(zip(X, Y))

    return data


dev = qml.device("default.qubit", wires=5)


@qml.qnode(dev, diff_method='backprop', wires=5)
def quantum_neural_network(x, w, depth):
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

    # variational circuit
    for j in range(9):
        for i in range(4):
            qml.RY(w[j][i], wires=i)
        qml.broadcast(qml.CNOT, wires=[0, 1, 2, 3], pattern="all_to_all", parameters=None, kwargs=None)

    for i in range(4):
        qml.RY(w[9][i], wires=i)

    dev.shots = 1000

    for i in range(4):
        qml.CNOT(wires=[i, 4])

    return qml.expval(qml.PauliZ(wires=4))


def get_parity_prediction(x, w, depth):
    np_measurements = (quantum_neural_network(x, w, depth) + 1.) / 2.

    return np.array([np_measurements, 1. - np_measurements])


def single_loss(w, x, y, depth):
    prediction = get_parity_prediction(x, w, depth)
    return rel_ent(prediction, y)


def rel_ent(pred, y):
    return -np.log(pred)[int(y)]


def average_loss(w, data, depth):
    c = 0
    Fisher = np.zeros((40, 40))
    for i, (x, y) in enumerate(data):
        single_cost = single_loss(w, x, y, depth)

        c += single_cost
        # print(single_cost)
        grad_fn = qml.grad(single_loss, argnum=0)
        gradient = grad_fn(w, x, y, depth).flatten()
        # print(gradient)
        # gradient = gradient/np.linalg.norm(gradient)
        Fisher += np.outer(gradient, gradient)

    return c / len(data), Fisher / len(data)



def get_all_fishers(n_iter, depth):
    all_fishers = np.zeros((n_iter, 40, 40))
    count = 0
    data = initialise_data()
    for i in range(n_iter):
        w = np.array(np.split(np.random.uniform(size=(40,), low=-1.0, high=1.0), 10), requires_grad=True)
        Fisher = average_loss(w, data, depth)[1]
        all_fishers[i] = Fisher
    return all_fishers


def normalise_fishers(Fishers):
    num_samples = len(Fishers)
    TrF_integral = (1 / num_samples) * np.sum([np.trace(F) for F in Fishers])
    return [((40) / TrF_integral) * F for F in Fishers]