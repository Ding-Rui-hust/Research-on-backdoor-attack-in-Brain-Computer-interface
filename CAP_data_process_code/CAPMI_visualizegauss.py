from scipy.signal import butter, lfilter, resample
from tqdm import tqdm
from pylab import genfromtxt
import scipy.io as io
import numpy as np
import pandas as pd
import lib.utils as utils
import random
import os
import sys
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA
sys.path.append('..')
from methods import pulse_noise
from scipy.io import loadmat

npp_params=[0.3,5,0.1]
SNR=0.1

path='EEG_Data/MI_DR02/'
data = loadmat(path + 'guasspoiplot-{}-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2],SNR))
x_guasspoi = data['x_guasspoi']
x_signal = data['x_signal']

x_signal=np.squeeze(x_signal)
x_guasspoi=np.squeeze(x_guasspoi)



x_c=x_signal[0]
x_p=x_guasspoi[0]


import numpy as np
import matplotlib.pyplot as plt

# max_, min_ = np.max(x_c), np.min(x_c)
# x_c = (x_c - min_) / (max_ - min_)
# x_p = (x_p - min_) / (max_ - min_)

fig, ax1= plt.subplots(1, sharey=True, figsize=(24, 16))


s1 = np.arange(x_c.shape[1]) * 1.0 / 128
#am = ams[data_name]

linewidth = 3.0
fontsize = 24
l3, = ax1.plot(s1, x_p[0] , linewidth=linewidth, color='red')  # plot adv data
l4, = ax1.plot(s1, x_c[0] , linewidth=linewidth, color='dodgerblue')  # plot clean data，先画第一行
for i in range(1,16):
    ax1.plot(s1, x_p[i] + 4*i, linewidth=linewidth, color='red')  # plot adv data
    ax1.plot(s1, x_c[i] + 4*i , linewidth=linewidth, color='dodgerblue')  # plot clean data 逐步向上画四行

ax1.set_xlabel('Time (s)', fontsize=fontsize)
ax1.set_title('After preprocessing', fontsize=fontsize + 2)

plt.ylim([-3.0, 64.0])
temp_y = np.arange(16)
#y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]

y_names = ['P3','C3','F3','Fz','F4','C4','P4','Cz','T3','T5','O1','O2','F7','F8','T6','T4']
plt.yticks(temp_y*4, y_names, fontsize=fontsize)#纵标签
plt.xticks(fontsize=fontsize)
plt.legend(handles=[l4, l3], labels=['Original sample', 'Gaussian sample'], loc='upper right', ncol=2,
          fontsize=fontsize - 2)#小标签

plt.subplots_adjust(wspace=1.0, hspace=1.0)
plt.tight_layout()
plt.show()