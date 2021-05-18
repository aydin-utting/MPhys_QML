#%%
import matplotlib.pyplot as plt
from pennylane import numpy as np
import matplotlib
import pickle as pk
font = {'family': 'serif', 'size': 12}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

l = np.load("../data/gaussian_losses_all.npy")
# l : ndarray shape (6,  2,  3,  2,  100)
#                  depth var rep t/l epoch
d = pk.load(open("../data/depth_effective_dimensions_gauss.data","rb"))

ryry_d_mean = np.array([np.mean(d["RYRY"][k]) for k in d["RYRY"]])
ryry_d_std = np.array([np.std(d["RYRY"][k]) for k in d["RYRY"]])

ryrx_d_mean = np.array([np.mean(d["RYRX"][k]) for k in d["RYRX"] ])
ryrx_d_std = np.array([np.std(d["RYRX"][k]) for k in d["RYRX"] ])

ryry_means = l[:,0].mean(axis = 1)
ryry_stds = l[:,0].std(axis=1)
l[:,0,:,0,:] = -1.* l[:,0,:,0,:]
ryry_gaps = l[:,0].sum(axis=2).mean(axis=1)
ryry_gaps_stds = l[:,0].sum(axis=2).std(axis=1)
l[:,0,:,0,:] = -1.* l[:,0,:,0,:]

ryrx_means = l[:,1].mean(axis=1)
ryrx_stds = l[:,1].std(axis=1)

l[:,1,:,0,:] = -1.* l[:,1,:,0,:]
ryrx_gaps = l[:,1].sum(axis=2).mean(axis=1)
ryrx_gaps_stds = l[:,1].sum(axis=2).std(axis=1)
l[:,1,:,0,:] = -1.* l[:,1,:,0,:]



#%% Pearsons rank
from scipy.stats import pearsonr


ds = np.array([ 7.13958131, 32.06737627, 37.7815372 , 39.82330634, 39.84919044,
        40.03460219, 40.28987359, 25.75199123, 33.81985753, 37.09150025,
        37.29457476, 37.4047557 , 37.80657033])
gaps = np.array([0.13      , 0.12645322, 0.2171434 , 0.26411669, 0.24387065,
        0.2024591 , 0.25092897, 0.2544878 , 0.22820458, 0.22303191,
        0.22299387, 0.23066517, 0.19662717])
pearsonr(ds,gaps)

#%%

"""fig,ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(6)
#ax[0].set_ylim(0.1,0.8)
for i in range(6):
    ax.errorbar(ryry_d_mean[i],ryry_gaps[i,-1],label=f"RYRY D{i}",xerr=ryry_d_std[i],yerr=ryry_gaps_stds[i,-1],marker="^",color=plt.cm.jet(i*(1/6)))
for i in range(6):
    ax.errorbar(ryrx_d_mean[i],ryrx_gaps[i,-1],yerr=ryrx_gaps_stds[i,-1],xerr=ryrx_d_std[i], label=f"RYRX D{i}",marker='o',color=plt.cm.jet(i*(1/6)))
ax.set_ylabel("Difference between training and validation loss")
ax.set_xlabel("Effective Dimension")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title(r"\bf{Validation training error gap against effective dimension}")
fig.savefig("../figures/gauss_gap_vs_deff",dpi=200)
fig.show()"""

#%%


fig,ax = plt.subplots()
fig.set_figwidth(8)
fig.set_figheight(6)
ax.errorbar(range(6),ryry_d_mean,yerr = ryry_d_std,label="RYRY",marker = 'o',color='black',linestyle="")
ax.errorbar(range(6),ryrx_d_mean,yerr=ryrx_d_std,label="RYRX",marker='s',color='green',linestyle="")
ax.legend()
ax.grid()
ax.set_title("Effective Dimension against depth for Gaussian clouds")
ax.set_xlabel("Encoding depth")
ax.set_ylabel("Effective Dimension")
fig.savefig("../figures/deff_vs_depth_gauss.png",dpi=200)
fig.show()

#%%
fig,ax = plt.subplots(1,2)
fig.set_figwidth(14)
fig.set_figheight(6)
#ax[0].set_ylim(0.1,0.8)
for i in range(6):
    ax[0].plot(range(100), ryry_means[i,0],label=f"RYRY D{i}")
    ax[0].fill_between(range(100),ryry_means[i,0]-ryry_stds[i,0],ryry_means[i,0]+ryry_stds[i,0],alpha=0.2)
    ax[1].plot(range(100), ryrx_means[i, 0], label=f"RYRX D{i}")
    ax[1].fill_between(range(100), ryrx_means[i, 0] - ryrx_stds[i, 0], ryrx_means[i, 0] + ryrx_stds[i, 0], alpha=0.2)
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("(a) RYRY Loss curves for Gaussian clouds")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].set_title("(b) RYRX Loss curves for Gaussian clouds")
ax[0].legend()
ax[1].legend()
fig.savefig("../figures/gaussian_loss_curves_qnn.png")
fig.show()

