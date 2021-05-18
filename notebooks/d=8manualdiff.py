import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd

def LogisticSigmoid(x):
  return 1./(1.+np.exp(-1.*x))

def dbydg(X,W,i):
  w,x,y,z = X[0],X[1],X[2],X[3]
  a,b,c,d,e,f,g = W[0],W[1],W[2],W[3],W[4],W[5],W[6+i]
  D = LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z)))*(1 - LogisticSigmoid(g*LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z)))))*LogisticSigmoid(g*LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z))))
  return D

def dbydf(X,W,i):
  w,x,y,z = X[0],X[1],X[2],X[3]
  a,b,c,d,e,f,g = W[0],W[1],W[2],W[3],W[4],W[5],W[6+i]
  D = g*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z))*(1 - LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z))))*LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z)))*(1 - LogisticSigmoid(g*LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z)))))*LogisticSigmoid(g*LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z))))
  return D

def dbyde(X,W,i):
  w,x,y,z = X[0],X[1],X[2],X[3]
  a,b,c,d,e,f,g = W[0],W[1],W[2],W[3],W[4],W[5],W[6+i]
  D = f*g*LogisticSigmoid(a*w + b*x + c*y + d*z)*(1 - LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z)))*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z))*(1 - LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z))))*LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z)))*(1 - LogisticSigmoid(g*LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z)))))*LogisticSigmoid(g*LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z))))
  return D

def dbydabcd(j,X,W,i):
  w,x,y,z = X[0],X[1],X[2],X[3]
  a,b,c,d,e,f,g = W[0],W[1],W[2],W[3],W[4],W[5],W[6+i]
  D = e*f*g*X[j]*(1 - LogisticSigmoid(a*w + b*x + c*y + d*z))*LogisticSigmoid(a*w + b*x + c*y + d*z)*(1 - LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z)))*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z))*(1 - LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z))))*LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z)))*(1 - LogisticSigmoid(g*LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z)))))*LogisticSigmoid(g*LogisticSigmoid(f*LogisticSigmoid(e*LogisticSigmoid(a*w + b*x + c*y + d*z))))
  return D

def F(X,W):
  w,x,y,z = X[0],X[1],X[2],X[3]
  a,b,c,d,e,f,g,h = W[0],W[1],W[2],W[3],W[4],W[5],W[6],W[7]
  return np.array([LogisticSigmoid(LogisticSigmoid(LogisticSigmoid(LogisticSigmoid(np.matmul(np.array([a, b, c, d]),np.array([w, x, y, z])))*e)*f)*g),
    LogisticSigmoid(LogisticSigmoid(LogisticSigmoid(LogisticSigmoid(np.matmul(np.array([a, b, c, d]),np.array([w, x, y, z])))*e)*f)*h)])

def cross_ent(X,W,Y):
  P = softmax(F(X,W))
  return -1.*np.log(P[Y])

def softmax(o,i=None):
  if i == None:
    return np.array([np.exp(o[i])/sum(np.exp(o)) for i in range(len(o))])
  else:
    return np.exp(o[i])/sum(np.exp(o))

def grad(X,W,Y):
  g = np.zeros(8)
  for i in range(4):
    g[i]= dbydabcd(i,X,W,Y)
  g[4] = dbyde(X,W,Y)
  g[5] = dbydf(X,W,Y)
  g[6+Y] = dbydg(X,W,Y)
  return (softmax(F(X,W),Y)-1)*np.array(g)

df = pd.read_csv('../data/iris.data',names=['a','b','c','d','species'])
df=df.query("species=='Iris-setosa' or species=='Iris-versicolor'")
df.replace('Iris-setosa',0,inplace=True)
df.replace('Iris-versicolor',1,inplace=True)
X = df.astype(float).values[:,0:4]
Y = df.values.astype(int)[:,4]

ww = np.random.uniform(-1.,1.,(8,))
for n in range(100):
  batch_g = 0.
  for i in range(100):
    batch_g += grad(X[i],ww,Y[i])
  ww += -0.1*0.01*batch_g
  loss = sum([cross_ent(X[j],ww,Y[j]) for j in range(100)])/100
  if n%10 == 0:
    print(f"Epoch: {n}, Loss: {loss}")

import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

class Net(nn.Module):
  def __init__(self):
    self.fc1 = nn.Linear(4,1,bias=False)
    self.fc2 = nn.Linear(1,1,bias=False)
    self.fc3 = nn.linear(1,1,bias=False)
    self.fc4 = nn.Linear(1,2,bias=False)
    nn.init.uniform_(self.fc1.weight, -1., 1.)
    nn.init.uniform_(self.fc2.weight, -1., 1.)
    nn.init.uniform_(self.fc3.weight, -1., 1.)
    nn.init.uniform_(self.fc4.weight, -1., 1.)
        
    self.sig = nn.Sigmoid

  def forward(self,x):
    x = self.sig(self.fc1(x))
    x = self.sig(self.fc2(x))
    x = self.sig(self.fc3(x))
    x = self.sig(self.fc4(x))
    return x


tX = Variable(torch.tensor(df.astype(float).values[:,0:4]).float())
tY = Variable(torch.tensor(df.values.astype(int)[:,4]).long())

# TRAIN
runs = 1
epochs = 100
epoch_loss=np.empty((runs,epochs))

all_fishers = []
all_w = []
for r in range(runs):
    net = nn.Linear(4, 2, bias=False)
    nn.init.uniform_(net.weight, -1., 1.)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    for epoch in range(epochs):
        opt.zero_grad()
        pred = net(tX)
        pred = F.softmax(pred)
        loss = F.cross_entropy(pred,tY)
        loss.backward()
        opt.step()
        epoch_loss[r,epoch]=loss.item()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

