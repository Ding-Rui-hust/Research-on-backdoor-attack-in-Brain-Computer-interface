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

def bandpass(sig, band, fs):
    B, A = butter(5, np.array(band) / (fs / 2), btype='bandpass')
    return lfilter(B, A, sig, axis=0)



sample_freq = 300.0
epoc_window = 2 * sample_freq#可以变

subjects = ['0003']
npp_params=[0.5,   5,   0.1]
data_file = 'EEG_Data/MI_DR/raw/data_{}_raw.csv'


y=np.squeeze(np.array([1,0,0,1,1,1,0,1,1,0,
           1,1,1,1,0,0,1,1,1,1,
           0,1,0,1,0,0,0,1,0,1]))
X_cl=[]
Y_cl=[]
X_po=[]
Y_po=[]
Ek_cl=0
Ek_po=0


for index in tqdm(range(len(subjects))):
    x = []
    e = []
    s = []
    clean=True

    file_name = data_file.format(subjects[index])
    sig = np.array(pd.read_csv(file_name).values)########读取CSV文件

    EEG = sig[:, 1:-2]#取每行从第二个到倒数第三个（去掉最后两个）
    Trigger = sig[:, -1]#取每行的最后一位

    idxFeedBack = np.where(Trigger == 1)[0]

    if not clean:
        npp = pulse_noise([1, 9, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                          proportion=npp_params[2])
        amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

        for _, idx in enumerate(idxFeedBack):
            idx = int(idx)
            EEG[idx:int(idx + epoc_window), :] = np.transpose(npp.squeeze() * amplitude,
                                                                  (1, 0)) + EEG[idx:int(idx + epoc_window), :]

    sig_F = bandpass(EEG, [8.0, 30.0], sample_freq)

    for _, idx in enumerate(idxFeedBack):
        idx = int(idx)
        s_sig = sig_F[idx:int(idx + epoc_window), :]
        s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
        x.append(s_sig)
        s.append(idx)


    x = np.array(x)
    x = np.transpose(x, (0, 2, 1))

    s = np.squeeze(np.array(s))
    y = np.squeeze(np.array(y))
    x_c = utils.standard_normalize(x)


for index in tqdm(range(len(subjects))):
    x = []
    e = []
    s = []
    clean=False

    file_name = data_file.format(subjects[index])
    sig = np.array(pd.read_csv(file_name).values)########读取CSV文件

    EEG = sig[:, 1:-2]#取每行从第二个到倒数第三个（去掉最后两个）
    Trigger = sig[:, -1]#取每行的最后一位

    idxFeedBack = np.where(Trigger == 1)[0]

    if not clean:
        npp = pulse_noise([1, 9, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                          proportion=npp_params[2])
        amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

        for _, idx in enumerate(idxFeedBack):
            idx = int(idx)
            EEG[idx:int(idx + epoc_window), :] = np.transpose(npp.squeeze() * amplitude,
                                                                  (1, 0)) + EEG[idx:int(idx + epoc_window), :]

    sig_F = bandpass(EEG, [8.0, 30.0], sample_freq)

    for _, idx in enumerate(idxFeedBack):
        idx = int(idx)
        s_sig = sig_F[idx:int(idx + epoc_window), :]
        s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
        x.append(s_sig)
        s.append(idx)


    x = np.array(x)
    x = np.transpose(x, (0, 2, 1))

    s = np.squeeze(np.array(s))
    y = np.squeeze(np.array(y))
    x_p = utils.standard_normalize(x)

# ica = UnsupervisedSpatialFilter(FastICA(9), average=False)
# # 进行PCA处理
# x_p = ica.fit_transform(x_p)
# x_c = ica.fit_transform(x_c)
#
# x_c=x_c[0]
# x_p=x_p[0]


import numpy as np
import matplotlib.pyplot as plt
from lib.load_data import load
from scipy.io import loadmat
import lib.visualization as vis

max_, min_ = np.max(x_c), np.min(x_c)
x_c = (x_c - min_) / (max_ - min_)
x_p = (x_p - min_) / (max_ - min_)

fig, ax1= plt.subplots(1, sharey=True, figsize=(7.5, 3.5))


s1 = np.arange(x_c.shape[1]) * 1.0 / 128
#am = ams[data_name]

linewidth = 1.0
fontsize = 10
l3, = ax1.plot(s1, x_p[0] - np.mean(x_p[0]), linewidth=linewidth, color='red')  # plot adv data
l4, = ax1.plot(s1, x_c[0] - np.mean(x_p[0]), linewidth=linewidth, color='dodgerblue')  # plot clean data，先画第一行
for i in range(1, 5):
    ax1.plot(s1, x_p[i] + i - np.mean(x_p[i]), linewidth=linewidth, color='red')  # plot adv data
    ax1.plot(s1, x_c[i] + i - np.mean(x_p[i]), linewidth=linewidth, color='dodgerblue')  # plot clean data 逐步向上画四行

ax1.set_xlabel('Time (s)', fontsize=fontsize)
ax1.set_title('After preprocessing', fontsize=fontsize + 2)

plt.ylim([-0.5, 5.0])
temp_y = np.arange(5)
#y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]

y_names = ['P3','C3','F3','Fz','F4','C4','P4','Cz','T3','T5','O1','O2','F7','F8','T6','T4']
plt.yticks(temp_y, y_names, fontsize=fontsize)#纵标签
plt.legend(handles=[l4, l3], labels=['Original sample', 'Poisoned sample'], loc='upper right', ncol=2,
          fontsize=fontsize - 2)#小标签

plt.subplots_adjust(wspace=1.0, hspace=1.0)
plt.tight_layout()
#plt.savefig('fig/signal_example.png', dpi=300)
plt.show()