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

sys.path.append('..')
from methods import pulse_noise


def bandpass(sig, band, fs):
    B, A = butter(5, np.array(band) / (fs / 2), btype='bandpass')
    return lfilter(B, A, sig, axis=0)



sample_freq = 200.0
epoc_window = 1.3 * sample_freq

subjects = ['02', '06', '07', 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
npp_params=[0.15,   5,   0.1]
#subjects = ['02', '06', '07', 11, 12, 13, 14, 16]#只取了一半
data_file = 'EEG_Data/ERN/raw/Data_S{}_Sess0{}.csv'


y = genfromtxt('EEG_Data/ERN/raw/TrainLabels.csv', delimiter=',', skip_header=1)[:, 1]
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
    for sess in range(5):#可以减少一部分，原来是5
        sess = sess + 1
        file_name = data_file.format(subjects[index], sess)
        sig = np.array(pd.read_csv(file_name).values)########读取CSV文件

        EEG = sig[:, 1:-2]
        Trigger = sig[:, -1]

        idxFeedBack = np.where(Trigger == 1)[0]

        if not clean:
            npp = pulse_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                              proportion=npp_params[2])
            amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

            for _, idx in enumerate(idxFeedBack):
                idx = int(idx)
                EEG[idx:int(idx + epoc_window), :] = np.transpose(npp.squeeze() * amplitude,
                                                                      (1, 0)) + EEG[idx:int(idx + epoc_window), :]

        sig_F = bandpass(EEG, [1.0, 40.0], sample_freq)

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
    x = utils.standard_normalize(x)

    if Ek_cl == 0:  # 解决concatenate无法拼接空数组的问题
        X_cl = x
        Y_cl = y[index * 340:(index + 1) * 340]
        Ek_cl = 1
    else:
        X_cl = np.concatenate((X_cl, x), axis=0)
        Y_cl = np.concatenate((Y_cl, y[index * 340:(index + 1) * 340]), axis=0)

x1 = X_cl[np.where(Y_cl == 0)]  # 消除类别不平衡的问题
x2 = X_cl[np.where(Y_cl == 1)]
sample_num = min(len(x1), len(x2))
idx1, idx2 = utils.shuffle_data(len(x1)), utils.shuffle_data(len(x2))
X_cl = np.concatenate([x1[idx1[:sample_num]], x2[idx2[:sample_num]]], axis=0)
Y_cl = np.concatenate([np.zeros(shape=[sample_num]), np.ones(shape=[sample_num])], axis=0)



for index in tqdm(range(len(subjects))):
    x = []
    e = []
    s = []
    clean=False
    for sess in range(5):#可以减少一部分，原来是5
        sess = sess + 1
        file_name = data_file.format(subjects[index], sess)
        sig = np.array(pd.read_csv(file_name).values)########读取CSV文件

        EEG = sig[:, 1:-2]
        Trigger = sig[:, -1]

        idxFeedBack = np.where(Trigger == 1)[0]

        if not clean:
            npp = pulse_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                              proportion=npp_params[2])
            amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

            for _, idx in enumerate(idxFeedBack):
                idx = int(idx)
                EEG[idx:int(idx + epoc_window), :] = np.transpose(npp.squeeze() * amplitude,
                                                                      (1, 0)) + EEG[idx:int(idx + epoc_window), :]

        sig_F = bandpass(EEG, [1.0, 40.0], sample_freq)

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
    x = utils.standard_normalize(x)

    if Ek_po == 0:  # 解决concatenate无法拼接空数组的问题
        X_po = x
        Y_po = y[index * 340:(index + 1) * 340]
        Ek_po= 1
    else:
        X_po = np.concatenate((X_po, x), axis=0)
        Y_po = np.concatenate((Y_po, y[index * 340:(index + 1) * 340]), axis=0)

x1 = X_po[np.where(Y_po == 0)]  # 消除类别不平衡的问题
x2 = X_po[np.where(Y_po == 1)]
sample_num = min(len(x1), len(x2))
#idx1, idx2 = utils.shuffle_data(len(x1)), utils.shuffle_data(len(x2))
X_po = np.concatenate([x1[idx1[:sample_num]], x2[idx2[:sample_num]]], axis=0)
Y_po = np.concatenate([np.zeros(shape=[sample_num]), np.ones(shape=[sample_num])], axis=0)


X_cl=X_cl[:, np.newaxis, :, :]
X_po=X_po[:, np.newaxis, :, :]
leng=len(X_cl)
idx_al=np.arange(leng)#


idx_cl,_, idx_po, _ = utils.split_data([idx_al, idx_al], split=0.86, shuffle=True)
idx_po,_,idx_test_po,_=utils.split_data([idx_po, idx_po], split=0.5, shuffle=True)
x_train=X_cl[idx_cl]
y_train=Y_cl[idx_cl]
x_poison=X_po[idx_po]
y_poison=Y_po[idx_po]
x_test=X_cl[idx_test_po]
y_test=Y_cl[idx_test_po]
x_test_poison=X_po[idx_test_po]##
y_test_poison=Y_po[idx_test_po]##

x_train, y_train, x_validation, y_validation = utils.split_data([x_train, y_train], split=0.8, shuffle=True)
save_dir = 'EEG_Data/ERN/'
save_file = save_dir + 'data2-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2])

io.savemat(save_file, {'x_train': x_train,'y_train': y_train, 'x_validation':x_validation,'y_validation':y_validation,
                       'x_poison': x_poison,'y_poison':y_poison,'x_test':x_test,'y_test':y_test ,
                       'x_test_poison':x_test_poison,'y_test_poison':y_test_poison})