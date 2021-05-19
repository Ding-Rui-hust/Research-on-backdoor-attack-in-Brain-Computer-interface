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

npp_params=[0.2,5,0.1]


path='EEG_Data/MI_DR02/'
data = loadmat(path + 'data2-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2]))
x_train = data['x_train']
y_train = data['y_train']
x_validation = data['x_validation']
y_validation = data['y_validation']
x_poison = data['x_poison']
y_poison = data['y_poison']
x_test= data['x_test']
y_test = data['y_test']
x_test_poison = data['x_test_poison']
y_test_poison = data['y_test_poison']



x_train=np.squeeze(x_train)
x_validation=np.squeeze(x_validation)
x_poison=np.squeeze(x_poison)
x_test=np.squeeze(x_test)
x_test_poison=np.squeeze(x_test_poison)
y_train=np.squeeze(y_train)
y_validation=np.squeeze(y_validation)
y_poison=np.squeeze(y_poison)
y_test=np.squeeze(y_test)
y_test_poison=np.squeeze(y_test_poison)

# path='EEG_Data/MI_DR02/'
# data = loadmat(path + 'dataPOI0.2.mat')
# x_POI = data['x_POI']
# y_POI = data['y_POI']
# x_POI=np.squeeze(x_POI)
# y_POI=np.squeeze(y_POI)

# ica = UnsupervisedSpatialFilter(FastICA(9), average=False)
# # 进行PCA处理
# x_p = ica.fit_transform(x_p)
# x_c = ica.fit_transform(x_c)



x_c=x_test[0]
x_p=x_test_poison[0]


import numpy as np
import matplotlib.pyplot as plt
from lib.load_data import load
from scipy.io import loadmat
import lib.visualization as vis

max_, min_ = np.max(x_c), np.min(x_c)
x_c = (x_c - min_) / (max_ - min_)
x_p = (x_p - min_) / (max_ - min_)

fig, ax1= plt.subplots(1, sharey=True, figsize=(24, 16))


s1 = np.arange(x_c.shape[1]) * 1.0 / 128
#am = ams[data_name]

linewidth = 3.0
fontsize = 24
l3, = ax1.plot(s1, x_p[0] - np.mean(x_p[0]), linewidth=linewidth, color='red')  # plot adv data
l4, = ax1.plot(s1, x_c[0] - np.mean(x_p[0]), linewidth=linewidth, color='dodgerblue')  # plot clean data，先画第一行
for i in range(1,16):
    ax1.plot(s1, x_p[i] + i - np.mean(x_p[i]), linewidth=linewidth, color='red')  # plot adv data
    ax1.plot(s1, x_c[i] + i - np.mean(x_p[i]), linewidth=linewidth, color='dodgerblue')  # plot clean data 逐步向上画四行

ax1.set_xlabel('Time (s)', fontsize=fontsize)
ax1.set_title('After preprocessing', fontsize=fontsize + 2)

plt.ylim([-0.5, 16.0])
temp_y = np.arange(16)
#y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]

y_names = ['P3','C3','F3','Fz','F4','C4','P4','Cz','T3','T5','O1','O2','F7','F8','T6','T4']
plt.yticks(temp_y, y_names, fontsize=fontsize)#纵标签
plt.xticks(fontsize=fontsize)
plt.legend(handles=[l4, l3], labels=['Original sample', 'Poisoned sample'], loc='upper right', ncol=2,
          fontsize=fontsize - 2)#小标签

plt.subplots_adjust(wspace=1.0, hspace=1.0)
plt.tight_layout()
#plt.savefig('fig/signal_example.png', dpi=300)
plt.show()