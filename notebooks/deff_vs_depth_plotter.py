import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
deffs = pkl.load(open("../data/depth_effective_dimensions_google.data","rb"))
deffs_google = pkl.load(open("../data/depth_effective_dimensions.data","rb"))
import matplotlib

font = {'family': 'serif', 'size': 18}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

for k in deffs_google:
    for d in deffs_google[k]:
        for i in range(3):
            deffs_google[k][d][i+2] = deffs[k][d][i]

for k in deffs_google:
    for d in deffs_google[k]:
        deffs_google[k][d].sort(reverse=True)

for k in deffs_google:
    for d in deffs_google[k]:
        for i in [4,3,2,1,0]:
            if deffs_google[k][d][i] == 0.0:
                deffs_google[k][d].pop(i)

mean_deffs = {k : np.zeros(6) for k in deffs_google}
std_deffs = {k : np.zeros(6) for k in deffs_google}
for k in deffs_google:
    for d in deffs_google[k]:
        mean_deffs[k][d] = np.array(deffs_google[k][d]).mean()/8.
        std_deffs[k][d] = np.array(deffs_google[k][d]).std()/8.

fig3,ax3 = plt.subplots()
fig3.set_figwidth(10.)
fig3.set_figheight(8.)
for k in mean_deffs:
    ax3.errorbar(range(6),mean_deffs[k],yerr=std_deffs[k])
ax3.grid()
ax3.legend(mean_deffs.keys())
ax3.set_title(r"\bf{Effective Dimension against encoding depth}")
ax3.set_ylabel(r"Normalised Effective Dimension $\frac{d_{1,n}}{d}$")
ax3.set_xlabel("Encoding depth")
fig3.show()

qnn_loss_rx = np.load("../data/qnn_loss_depths_RYRX.npy")
ryrx_mins = np.zeros(6)
ryrx_stds = np.zeros(6)
for i in range(6):
    ryrx_mins[i] = np.mean(np.min(qnn_loss_rx[i], axis=1))
    ryrx_stds[i] = np.std(np.min(qnn_loss_rx[i], axis=1))

qnn_loss_ry0123 = np.load("../data/qnn_loss_RYRY_0&1&2&3.npy")
qnn_loss_ry45 = np.load("../data/qnn_RYRY_4&5.npy")

ryry_mins = np.zeros(6)
ryry_stds = np.zeros(6)
for i in range(4):
    ryry_mins[i] = np.mean(np.min(qnn_loss_ry0123[i], axis=1))
    ryry_stds[i] = np.std(np.min(qnn_loss_ry0123[i], axis=1))
for i in range(2):
    ryry_mins[i+4] = np.mean(np.min(qnn_loss_ry45[i], axis=1))
    ryry_stds[i+4] = np.std(np.min(qnn_loss_ry45[i], axis=1))

loss_mins = {'RYRY': ryry_mins, 'RYRX':  ryrx_mins}
loss_stds = {'RYRY': ryry_stds, 'RYRX':  ryrx_stds}

fig4,ax4 = plt.subplots()
fig4.set_figwidth(10.)
fig4.set_figheight(8.)
for k in mean_deffs:
    ax4.errorbar(mean_deffs[k],loss_mins[k],xerr=std_deffs[k],yerr=loss_stds[k],linestyle=" ")
ax4.grid()
ax4.legend(mean_deffs.keys())
ax4.set_title(r"\bf{Minimum Loss against Effective Dimension}")
ax4.set_ylabel("Minimum Training Loss")
ax4.set_xlabel(r"Normalised Effective Dimension $\frac{d_{1,n}}{d}$")
fig4.show()
'''
RYRY = np.array([ [deffs['RYRY'][d][0],deffs_google['RYRY'][d][0]] for d in range(6)] )

RYRX = np.array([ [deffs['RYRX'][d][0],deffs_google['RYRX'][d][0]] for d in range(6)] )



fig,ax = plt.subplots()
ax.plot(range(6),[deffs['RYRY'][d][0] for d in deffs['RYRY']],marker='o')

ax.plot(range(6),[deffs['RYRX'][d][0] for d in deffs['RYRX']],marker='o')

ax.set_ylabel("Effective Dimension")
ax.set_xlabel("Encoding depth")
ax.grid()
ax.legend(["RYRY","RYRX"])
fig.show()

fig2,ax2 = plt.subplots()
ax2.errorbar(range(6),RYRY.mean(axis=1),yerr=RYRY.std(axis=1))

ax2.errorbar(range(6),RYRX.mean(axis=1),yerr=RYRX.std(axis=1))
fig2.show()
'''