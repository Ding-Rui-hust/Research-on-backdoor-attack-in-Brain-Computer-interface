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
data = loadmat(path + 'dataPOI2_0.2.mat')
x_POI02 = data['x_POI']
y_POI02 = data['y_POI']
x_POI02=np.squeeze(x_POI02)
y_POI02=np.squeeze(y_POI02)

data = loadmat(path + 'dataPOI0.1.mat')
x_POI01 = data['x_POI']
y_POI01 = data['y_POI']
x_POI01=np.squeeze(x_POI01)
y_POI01=np.squeeze(y_POI01)

data = loadmat(path + 'dataPOI0.05.mat')
x_POI005 = data['x_POI']
y_POI005 = data['y_POI']
x_POI005=np.squeeze(x_POI005)
y_POI005=np.squeeze(y_POI005)




# ica = UnsupervisedSpatialFilter(FastICA(9), average=False)
# # 进行PCA处理
# x_p = ica.fit_transform(x_p)
# x_c = ica.fit_transform(x_c)



x_c=x_POI02[0]
x_d=x_POI01[0]
x_p=x_POI005[0]



import numpy as np
import matplotlib.pyplot as plt
from lib.load_data import load
from scipy.io import loadmat
import lib.visualization as vis

max_, min_ = np.max(x_c), np.min(x_c)
x_c = (x_c - min_) / (max_ - min_)
x_p = (x_p - min_) / (max_ - min_)
x_d = (x_d - min_) / (max_ - min_)

fig, ax1= plt.subplots(1, sharey=True, figsize=(7.5, 10))


s1 = np.arange(x_c.shape[1]) * 1.0 / 128
#am = ams[data_name]

linewidth = 1.0
fontsize = 10
l3, = ax1.plot(s1, x_p[0] - np.mean(x_p[0]), linewidth=linewidth, color='red')  # plot adv data
l4, = ax1.plot(s1, x_c[0] - np.mean(x_c[0]), linewidth=linewidth, color='dodgerblue')  # plot clean data，先画第一行
l5, = ax1.plot(s1, x_d[0] - np.mean(x_d[0]), linewidth=linewidth, color='green')
for i in range(1,16):
    ax1.plot(s1, x_p[i] + i - np.mean(x_p[i]), linewidth=linewidth, color='red')  # plot adv data
    ax1.plot(s1, x_c[i] + i - np.mean(x_c[i]), linewidth=linewidth, color='dodgerblue')  # plot clean data 逐步向上画四行
    ax1.plot(s1, x_d[i] + i - np.mean(x_d[i]), linewidth=linewidth, color='green')

ax1.set_xlabel('Time (s)', fontsize=fontsize)
ax1.set_title('After preprocessing', fontsize=fontsize + 2)

plt.ylim([-0.5, 16.0])
temp_y = np.arange(16)
y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
plt.yticks(temp_y, y_names, fontsize=fontsize)#纵标签
plt.legend(handles=[l4,l5,l3], labels=['0.2V POISON','0.1V POISON','0.05V POISON'], loc='upper right', ncol=3,
          fontsize=fontsize - 3)#小标签

plt.subplots_adjust(wspace=1.0, hspace=1.0)
plt.tight_layout()
#plt.savefig('fig/signal_example.png', dpi=300)
plt.show()