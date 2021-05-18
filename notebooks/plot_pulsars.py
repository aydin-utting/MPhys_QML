#%%
import matplotlib.pyplot as plt
from pennylane import numpy as np
import numpy
import matplotlib
import pickle as pk

font = {'family': 'serif', 'size': 12}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

l = np.load("../data/losses (3)")

ryry_means = l[:,0,0]
#ryry_stds = l[:,0].std(axis=1)
#l[:,0,:,0,:] = -1.* l[:,0,:,0,:]
ryry_gaps = l[:,0].sum(axis=2)[:,0].mean(axis=1)
#ryry_gaps_stds = l[:,0].sum(axis=2).std(axis=1)


ryrx_means = l[:,1,0]
#ryrx_stds = l[:,1].std(axis=1)

#l[:,1,:,0,:] = -1.* l[:,1,:,0,:]
ryrx_gaps = l[:,1].sum(axis=2)[:,0]

#%%
#ryrx_gaps_stds = l[:,1].sum(axis=2).std(axis=1)
for i in range(1,6):
    plt.plot(range(100),ryry_means[i,:,1],label=f"RYRY{i}")
    plt.plot(range(100),ryrx_means[i,:,1],label=f"RYRX{i}")
plt.legend()
plt.show()



#%%
d = pk.load(open("../data/depth_effective_dimensions_IRIS.data","rb"))
d2 = pk.load(open("../data/depth_effective_dimensions_pulsars_02pi.data","rb"))
deffs = {k : np.zeros((6,2)) for k in d}

markers = {"RYRY" : 'o',
            "RYRX" : 's'}
colors = {'RYRY' : 'black',
          "RYRX" : 'g'}
for k in d:
    for i in d[k]:
        deffs[k][i][0] = numpy.mean(d[k][i])
        deffs[k][i][1] = numpy.std(d[k][i])



for k in deffs:
    plt.errorbar(range(6),deffs[k][:,0], yerr=deffs[k][:,1],label=k,marker=markers[k],linestyle = "",color=colors[k])
#plt.hlines(y=16,xmin=-1,xmax=6,linestyle="--",color='r',alpha=0.5)

plt.xlim(-0.2,5.2)
plt.title("Effective Dimension against depth for IRIS")
plt.xlabel("Encoding depth")
plt.ylabel("Effective Dimension")
plt.legend()
plt.grid()
plt.savefig("../figures/depth_effd_IRIS.png",dpi=200)
plt.show()