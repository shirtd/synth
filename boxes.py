import matplotlib.pyplot as plt
from util.data import *
import pandas as pd
plt.ion()

# plt.figure(1,figsize=(18,9))
# ax = [plt.subplot(611+i) for i in range(6)]
# fig, ax = plt.subplots(2,3, sharex=True, sharey=True, figsize=(10,8))
# iax = {-10: ax[0,0], -6: ax[0,1], -2: ax[0,2], 2: ax[1,2], 6: ax[1,1], 10: ax[1,0]}
fig, ax = plt.subplots(1,6, sharex=True, sharey=True, figsize=(16,8))
iax = {-10: ax[0], -6: ax[1], -2: ax[2], 2: ax[3], 6: ax[4], 10: ax[5]}

fig.tight_layout()
fig.subplots_adjust(wspace=0)

FILE = 'data/chunk7_projection_dot.csv'
df = pd.read_csv(FILE)

dfm = {mod : df.loc[df['MOD'] == mod] for mod in MODS}
dfs = {mod : {snr : dfm[mod].loc[dfm[mod]['SNR'] == snr].drop(ICOLS,axis=1).values for snr in SNRS} for mod in MODS}

def plot_mod(col):
    for snr in SNRS:
        iax[snr].cla()
        iax[snr].boxplot([dfs[mod][snr][:,col] for mod in MODS],vert=False)
    plt.show(block=False)

for col in range(df.shape[1]-27):
    plot_mod(col)
    raw_input('column %d: %s' % (col,df.columns[27+col]))
