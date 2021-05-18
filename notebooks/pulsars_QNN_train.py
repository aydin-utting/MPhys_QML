

#Importing Libraries
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn import preprocessing

#Torch for Classical NN
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

#PennyLane for QNN
import pennylane as qml
from pennylane.optimize import AdamOptimizer

from tqdm.auto import tqdm


#Seed - DO NOT CHANGE
seed = 13

#number of training samples per batch
n_train = 100

#how many validation batches
n_batch_val = 1

# load PULSARS dataset
dataset = pd.read_csv('../data/HTRU_2.csv')
for k in dataset[dataset.columns[0:8]]:
    dataset[k] = minmax_scale(dataset[k], feature_range=(0.,2.*np.pi))
data0 = dataset[dataset[dataset.columns[8]]==0]
data0 = data0.sample(n=int((n_batch_val+1)*n_train/2),random_state=seed)
X0 = data0[data0.columns[0:8]].values
Y0 = data0[data0.columns[8]].values

data1 = dataset[dataset[dataset.columns[8]]==1]
data1 = data1.sample(n=int((n_batch_val+1)*n_train/2),random_state=seed)
X1 = data1[data1.columns[0:8]].values
Y1 = data1[data1.columns[8]].values

X = np.append(X0,X1,axis=0)
Y = np.append(Y0,Y1,axis=0)





#Separate the test and training datasets
train_X, validation_X, train_Y, validation_Y = train_test_split(X,Y, test_size=n_batch_val/(n_batch_val+1),random_state=seed)

all_validation_X = np.zeros((n_batch_val,n_train,8))
all_validation_Y = np.zeros((n_batch_val,n_train))

n = n_batch_val
#Split the validation dataset in batches
for i in range(n_batch_val-1):
    n-=1
    all_validation_X[i],validation_X,all_validation_Y[i],validation_Y = train_test_split(validation_X,validation_Y, test_size=n/(n+1),random_state=seed)

all_validation_X[n_batch_val-1] = validation_X
all_validation_Y[n_batch_val-1] = validation_Y





num_epochs = 100
loss_curve_x = np.array(range(num_epochs))




##----------------------------------------CLASSICAL NN ----------------------------------------------
def classical_train(train_X,train_Y,all_validation_X=None,all_validation_Y=None): 
    
    from torch.autograd import Variable
    
    class Net(nn.Module):
        # define nn
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(8,2,bias=False)
            nn.init.uniform_(self.fc1.weight,-1.,1)
            
        def forward(self, X):
            X = self.fc1(X)
            return X
        
    # wrap up with Variable in pytorch
    train_X = Variable(torch.Tensor(train_X).float())
    train_Y = Variable(torch.Tensor(np.float64(train_Y)).long())
    
    
    
    net = Net()
    
    criterion = nn.CrossEntropyLoss()# cross entropy loss
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    
    cnn_training_loss = np.zeros((num_epochs,1))
    cnn_validation_losses = np.zeros((num_epochs,all_validation_X.shape[0]))
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        out = net(train_X)
        loss = criterion(out,train_Y)
        loss.backward()
        cnn_training_loss[epoch][0] = loss.item()      
        #validation loss
        with torch.no_grad():
            validation_losses = np.array([])
            for i,validation_X in enumerate(all_validation_X):
                
                validation_X = Variable(torch.Tensor(validation_X).float())
                validation_out = net(validation_X)
                validation_loss = criterion(validation_out,Variable(torch.Tensor(np.float64(all_validation_Y[i])).long()))
                validation_losses = np.append(validation_loss.item(),validation_losses)   
                
            cnn_validation_losses[epoch] = validation_losses
            if epoch%(num_epochs-1)==0:
                print("Epoch = ",epoch, " Training Loss = ",loss.item()," Validation Loss Mean = ",np.mean(validation_losses)," Validation Loss Std = ",np.std(validation_losses))
        optimizer.step()     
    
    return np.append(cnn_training_loss,cnn_validation_losses,axis=1)

##---------------------------------------------------------------------------------------------------


#----------------------------------------QUANTUM NN-------------------------------------------------

def quantum_model_train(train_X,train_Y,all_validation_X=None,all_validation_Y=None,depth=2,variation="RYRY"):
    from pennylane import numpy as  np
    
    train_X = np.array(train_X,requires_grad=False)
    train_Y = np.array(train_Y,requires_grad=False)
    
    np_all_validation_X = np.zeros_like(all_validation_X)
    for i, validation_X in enumerate(all_validation_X):
        np_all_validation_X[i] = np.array(validation_X)
        
    np_all_validation_Y = np.zeros_like(all_validation_Y)
    for i, validation_Y in enumerate(all_validation_Y):
        np_all_validation_Y[i] = np.array(validation_Y)
        
    validation_X = np.array(validation_X,requires_grad=False)
    validation_Y = np.array(validation_Y,requires_grad=False)

    train_data = list(zip(train_X,train_Y))
    
    all_validation_data = [list(zip(all_validation_X[i],all_validation_Y[i])) for i in range(all_validation_X.shape[0])]
    
    dev = qml.device("default.qubit.autograd", wires=9) 

    
    @qml.qnode(dev, diff_method='backprop')
    def quantum_neural_network(x, w,depth=depth):
        
        #encoding circuit============================================================

        #Hadamards
        for i in range(8):
            qml.Hadamard(wires=i)

        #Accounting for depth=0
        if(depth==0):
            for i in range(8):
                qml.RZ(x[i],wires=i)

        #Multiple encoding layers:
        for k in range(depth):
            #RZ gates
            for i in range(8):
                qml.RZ(x[i],wires=i)
            #RZZ gates
            for i in range(8):
                for j in range(i):
                    qml.CNOT(wires=[j,i])
                    qml.RZ(((np.pi-x[i])*(np.pi-x[j])),wires=[i])
                    qml.CNOT(wires=[j,i])
                
        
        #variational circuit =======================================================
        for i in range(8):
            qml.RY(w[0][i],wires=i)
            
        qml.broadcast(qml.CNOT, wires=range(8), pattern="all_to_all", parameters=None, kwargs=None)
        
        for i in range(8):
            if variation=="RYRY":qml.RY(w[1][i],wires=i)
            elif variation=="RYRX":qml.RX(w[1][i],wires=i)
        
        for i in range(8):
          qml.CNOT(wires=[i,8])

        return qml.expval(qml.PauliZ(wires=8))
    
    def get_parity_prediction(x,w):
        
        np_measurements = (quantum_neural_network(x,w)+1.)/2.
        
        return np.array([np_measurements,1.-np_measurements])
    
    def average_loss(w, data):
        cost_value = 0.
        for i,(x, y) in enumerate(data):
           
            cost_value += single_loss(w,x,y)

        return cost_value/len(data)

    def single_loss(w,x,y):
        prediction = get_parity_prediction(x,w)
        #print(prediction[int(y)])
        return rel_ent(prediction, y)
    
    def rel_ent(pred,y):
        return -1.*np.log(pred)[int(y)]
    

    
    #initialise weights
    w = np.array(np.split(np.random.uniform(size=(16,),low=-1.,high=1.),2),requires_grad=True)
    ws = np.zeros((num_epochs,w.shape[0],w.shape[1]))
    learning_rate=0.1
    train_losses = np.zeros((num_epochs,1))
    validation_losses = np.zeros((num_epochs,all_validation_X.shape[0]))
    #Optimiser
    optimiser = AdamOptimizer(learning_rate)
    pbar = tqdm(range(num_epochs))
    for i in pbar:
        w, train_loss_value = optimiser.step_and_cost(lambda v: average_loss(v, train_data), w)
        ws[i] = w
      # w.requires_grad=False
      # val_losses = np.array([])
      # for validation_data in all_validation_data:
      #     validation_loss_value = average_loss(w, validation_data)
      #     val_losses= np.append(validation_loss_value,val_losses)
      # validation_losses[i] = val_losses
      # w.requires_grad=True
        train_losses[i][0] = train_loss_value
        pbar.set_postfix({"Train Loss" : train_loss_value})
      #if i%5==0:
       #   print("Epoch = ",i, " Training Loss = ",train_loss_value," Validation Loss = ",np.mean(val_losses)," Validation Loss Std = ",np.std(val_losses))
      
      
    
    return np.append(validation_losses,train_losses,axis=1),ws


n_iteration = 1
variations = ['RYRY']
depths=[2]
qnn_loss= np.zeros((len(depths),len(variations),n_iteration,num_epochs,n_batch_val+1))
ann_loss= np.zeros((n_iteration,num_epochs,n_batch_val+1))
weights = np.zeros((len(depths),len(variations),n_iteration,num_epochs,2,8))
for i in range(n_iteration):
    #ann_loss[i] = classical_train(train_X,train_Y,all_validation_X,all_validation_Y)
    for j,variation in enumerate(variations):
        for k,depth in enumerate(depths):
            print("depth= ",depth, " iteration= ", i, " variation=",variation)
            qnn_loss[k][j][i],weights[k][j][i] = quantum_model_train(train_X,train_Y,all_validation_X,all_validation_Y,depth=depth,variation=variation)
            np.save(open("../data/qnn_pulsars_train_test.data", "wb"), qnn_loss)
            
            