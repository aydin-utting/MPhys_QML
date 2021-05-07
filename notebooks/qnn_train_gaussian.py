import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from pennylane.optimize import AdamOptimizer

DIM = 40
W = 4
def initialise_data(n_samples):
    
    variance = 1
    X0 = np.array([[np.random.normal(loc=-1, scale=variance), 
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



dev = qml.device("default.qubit", wires=W+1)
@qml.qnode(dev, diff_method='backprop', wires=W+1)
def quantum_neural_network(x, w,depth,variation):
    
    dev.shots = 10000
    #encoding circuit===========================================================

    #Hadamards
    for i in range(W):
        qml.Hadamard(wires=i)
        
    #Accounting for depth=0
    if(depth==0):
        for i in range(W):
            qml.RZ(x[i],wires=i)

    #Multiple encoding layers:
    for k in range(depth):
        #RZ gates
        for i in range(W):
            qml.RZ(x[i],wires=i)
        #RZZ gates
        for i in range(W):
            for j in range(i):
                qml.CNOT(wires=[j,i])
                qml.RZ((np.pi-x[i])*(np.pi-x[j]),wires=[i])
                qml.CNOT(wires=[j,i])
    

    #variational circuit =======================================================
    for j in range(int(DIM/W)-1):
      for i in range(W):
          qml.RY(w[j][i],wires=i)
      qml.broadcast(qml.CNOT, wires=range(W), pattern="all_to_all", parameters=None, kwargs=None)

    for i in range(W):
        if variation=="RYRY":qml.RY(w[int(DIM/W)-1][i],wires=i)
        elif variation=="RYRX":qml.RX(w[int(DIM/W)-1][i],wires=i)
    
    for i in range(W):
      qml.CNOT(wires=[i,W])
    
    return qml.expval(qml.PauliZ(wires=W))


def get_parity_prediction(x,w,depth,variation):
    np_measurements = (quantum_neural_network(x,w,depth,variation)+1.)/2.
  
    return np.array([np_measurements,1.-np_measurements])

def single_loss(w,x,y,depth,variation):
    prediction = get_parity_prediction(x,w,depth,variation)
    return rel_ent(prediction, y)

def rel_ent(pred,y):
    return -np.log(pred)[int(y)]

def average_loss(w, data,depth,variation,want_fisher=False):
  c = 0
  Fisher = np.zeros((DIM,DIM))
  for i,(x, y) in enumerate(data):
    single_cost = single_loss(w,x,y,depth,variation)
    c += single_cost
    if(want_fisher):
        grad_fn = qml.grad(single_loss,argnum=0)
        gradient = grad_fn(w,x,y,depth,variation).flatten()
        Fisher += np.outer(gradient,gradient)
  if(want_fisher):
          return c/len(data),Fisher/len(data)
  else:
          return c/len(data)


def get_all_fishers(n_iter,depth,variation):
  all_fishers = np.zeros((n_iter,DIM,DIM))
  count = 0
  data = initialise_data()
  for i in range(n_iter):
    w = np.array(np.split(np.random.uniform(size=(DIM,),low=-1.0,high=1.0),2),requires_grad=True)
    Fisher = average_loss(w,data,depth,variation)[1]
    all_fishers[i]=Fisher
  return all_fishers

def normalise_fishers(Fishers):
    num_samples = len(Fishers)
    TrF_integral = (1 / num_samples) * np.sum([np.trace(F) for F in Fishers])
    return [((DIM) / TrF_integral) * F for F in Fishers]


def get_entropy(depth,variation,n_iter=5,bins=10):
  all_fishers = get_all_fishers(n_iter,depth,variation)
  normalised_fishers = normalise_fishers(all_fishers)
  EVs = np.array([])
  for F in normalised_fishers:
    val, v = np.linalg.eig(F)
    EVs = np.append(np.real(EVs),val)
  x, bins, p=plt.hist(EVs, bins=bins)
  entropy=0
  x = x/len(EVs)
  for i in x:
    if(i!=0):
      entropy-=i*np.log(i)
  return entropy

def train(train_data,validation_data,num_epochs,depth,variation):
    #initialise weights
    w = np.array(np.split(np.random.uniform(size=(DIM,),low=-1,high=1),int(DIM/W)),requires_grad=True)
    learning_rate=0.1
    train_losses = np.array([])
    validation_losses = np.array([])
     #Optimiser
    optimiser = AdamOptimizer(learning_rate)
    for i in range(num_epochs):
      w.requires_grad = False
      validation_loss_value = average_loss(w, validation_data,depth,variation)
      w.requires_grad = True
      validation_losses = np.append(validation_losses,validation_loss_value)
      w, train_loss_value = optimiser.step_and_cost(lambda v: average_loss(v, train_data,depth,variation), w)        
      if i%5==0:
          print("Epoch = ",i, " Training Loss = ",train_loss_value," Validation Loss= ",validation_loss_value)
      train_losses = np.append(train_losses,train_loss_value)
      
    
    return np.array([train_losses,validation_losses])


training_data = initialise_data(100)
validation_data = initialise_data(100)

num_epochs=100
n_iteration = 1
depths = [2]
variations = ['RYRY']
qnn_loss= np.zeros((len(depths),len(variations),n_iteration,2,num_epochs))
ann_loss= np.zeros((n_iteration,2,num_epochs))
for i in range(n_iteration):
    print("Classical train loss = ",np.min(ann_loss[i][0])," val loss = ",np.min(ann_loss[i][1]))
    for j,variation in enumerate(variations):
        for k,depth in enumerate(depths):
            print("depth = ",depth, " iteration = ", i, " variation =",variation)
            qnn_loss[k][j][i] = train(training_data,validation_data,num_epochs,depth=depth,variation=variation)
            np.save(open("./gaussian_train_losses","wb"),qnn_loss)
