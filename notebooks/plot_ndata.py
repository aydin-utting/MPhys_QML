
import matplotlib.pyplot as plt
from pennylane import numpy as np
import numpy
import matplotlib
import pickle as pk

font = {'family': 'serif', 'size': 12}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

d_2 = pk.load(open("../data/depth_effective_dimensions_gauss_NDATA.data","rb"))
d_5 = pk.load(open("../data/depth_effective_dimensions_gauss_NDATA_5.data","rb"))
d_c = pk.load(open("../data/classical_efs.data","rb"))
d_c[250] = 6.737085174020994
n_datas = [10, 20, 50, 100, 250, 500, 750, 1000]

fig,ax = plt.subplots()
#fig.set_figwidth(8)
for i in range(2,len(n_datas)-2):
    l2 = ax.scatter(n_datas[i], d_2['RYRX'][2][i],color='green')
    l5 = ax.scatter(n_datas[i], d_5['RYRX'][5][i],color='blue')

#for n in n_datas[2:-2]:
#    lc = plt.scatter(n, d_c[n],color='red')
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax.set_xticks(range(0,1100,100))
ax.set_xlabel("Number of data points")
ax.set_ylabel("Effective Dimension")
ax.grid()
#ax.legend([l2,l5,lc],["RYRX Depth 2","RYRX Depth 5","Classical"],loc='center left', bbox_to_anchor=(1, 0.5))
fig.savefig("../figures/NDATA_inset.png",dpi=200)
fig.show()