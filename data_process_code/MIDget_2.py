from scipy.signal import butter, lfilter, resample
from tqdm import tqdm
import scipy.io as io
import numpy as np
import lib.utils as utils
import random
import os
import sys

sys.path.append('..')

from methods import pulse_noise


def bandpass(sig, band, fs):
    B, A = butter(5, np.array(band) / (fs / 2), btype='bandpass')
    return lfilter(B, A, sig, axis=0)



sample_freq = 512.0
epoc_window = 1.75 * sample_freq#取的1.75s
start_time = 2.5

subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', 10, 11, 12, 13, 14]#为什么后面不加引号？
data_file = 'EEG_Data/MI/raw/'


file1 = 'S{}E.mat'
file2 = 'S{}T.mat'

# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
npp_params=[1,   5,   0.1]
X_cl=[]
Y_cl=[]
X_po=[]
Y_po=[]
Ek_cl=0
Ek_po=0
for s in tqdm(range(len(subjects))):
    x = []
    e = []
    labels = []
    clean=True

    data = io.loadmat(data_file + file1.format(subjects[s]))
    for i in range(3):#这是由数据集本身决定，1行3列，SE，下面一行5列ST，这里是把所有的数据conceret起来
        s_data = data['data'][0][i]
        EEG, trial, y = s_data['X'][0][0], s_data['trial'][0][0], s_data['y'][0][0]
        trial, y = trial.squeeze(), y.squeeze() - 1
        labels.append(y)

        if not clean:

            npp = pulse_noise([1, 15, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                              proportion=npp_params[2])
            amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]#计算标准差

            for _, idx in enumerate(trial):
                idx = int(idx)
                EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window),
                :] = np.transpose(
                    npp.squeeze() * amplitude,
                    (1, 0)) + EEG[
                              int(idx + start_time * sample_freq):int(
                                  idx + start_time * sample_freq + epoc_window),
                              :]

        sig_F = bandpass(EEG, [8.0, 30.0], sample_freq)

        for _, idx in enumerate(trial):
            idx = int(idx)
            s_EEG = EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :]
            s_sig = sig_F[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :]

            s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
            e.append(s_EEG)
            x.append(s_sig)

    data = io.loadmat(data_file + file2.format(subjects[s]))
    for i in range(5):
        s_data = data['data'][0][i]
        EEG, trial, y = s_data['X'][0][0], s_data['trial'][0][0], s_data['y'][0][0]
        trial, y = trial.squeeze(), y.squeeze() - 1
        labels.append(y)

        if not clean:
            npp = pulse_noise([1, 15, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                              proportion=npp_params[2])
            amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

            for _, idx in enumerate(trial):
                idx = int(idx)
                EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window),
                :] = np.transpose(
                    npp.squeeze() * amplitude,
                    (1, 0)) + EEG[
                              int(idx + start_time * sample_freq):int(
                                  idx + start_time * sample_freq + epoc_window),
                              :]

        sig_F = bandpass(EEG, [8.0, 30.0], sample_freq)

        for _, idx in enumerate(trial):
            idx = int(idx)
            s_EEG = EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :]
            s_sig = sig_F[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :]

            s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
            e.append(s_EEG)
            x.append(s_sig)

    e = np.array(e)
    e = np.transpose(e, (0, 2, 1))
    x = np.array(x)
    x = np.transpose(x, (0, 2, 1))

    s = np.squeeze(np.array(s))
    labels = np.squeeze(np.array(labels))
    labels=labels.flatten()
    e = utils.standard_normalize(e)
    x = utils.standard_normalize(x)

    # io.savemat(save_file.format(s), {'eeg': e[:, np.newaxis, :, :],
    #                                  'x': x[:, np.newaxis, :, :], 'y': labels})

    if Ek_cl==0:#解决concatenate无法拼接空数组的问题
        X_cl=x
        Y_cl=labels
        Ek_cl=1
    else:
        X_cl= np.concatenate((X_cl, x), axis=0)
        Y_cl= np.concatenate((Y_cl, labels), axis=0)


for s in tqdm(range(len(subjects))):
    x = []
    e = []
    labels = []
    clean = False

    data = io.loadmat(data_file + file1.format(subjects[s]))
    for i in range(3):  # 这是由数据集本身决定，1行3列，sE，下面一行5列ST，这里是把所有的数据conceret起来
        s_data = data['data'][0][i]
        EEG, trial, y = s_data['X'][0][0], s_data['trial'][0][0], s_data['y'][0][0]
        trial, y = trial.squeeze(), y.squeeze() - 1
        labels.append(y)

        if not clean:

            npp = pulse_noise([1, 15, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                              proportion=npp_params[2])
            amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]  # 计算标准差

            for _, idx in enumerate(trial):
                idx = int(idx)
                EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window),
                :] = np.transpose(
                    npp.squeeze() * amplitude,
                    (1, 0)) + EEG[
                              int(idx + start_time * sample_freq):int(
                                  idx + start_time * sample_freq + epoc_window),
                              :]

        sig_F = bandpass(EEG, [8.0, 30.0], sample_freq)

        for _, idx in enumerate(trial):
            idx = int(idx)
            s_EEG = EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :]
            s_sig = sig_F[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :]

            s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
            e.append(s_EEG)
            x.append(s_sig)

    data = io.loadmat(data_file + file2.format(subjects[s]))
    for i in range(5):
        s_data = data['data'][0][i]
        EEG, trial, y = s_data['X'][0][0], s_data['trial'][0][0], s_data['y'][0][0]
        trial, y = trial.squeeze(), y.squeeze() - 1
        labels.append(y)

        if not clean:
            npp = pulse_noise([1, 15, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                              proportion=npp_params[2])
            amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

            for _, idx in enumerate(trial):
                idx = int(idx)
                EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window),
                :] = np.transpose(
                    npp.squeeze() * amplitude,
                    (1, 0)) + EEG[
                              int(idx + start_time * sample_freq):int(
                                  idx + start_time * sample_freq + epoc_window),
                              :]

        sig_F = bandpass(EEG, [8.0, 30.0], sample_freq)

        for _, idx in enumerate(trial):
            idx = int(idx)
            s_EEG = EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :]
            s_sig = sig_F[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :]

            s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
            e.append(s_EEG)
            x.append(s_sig)

    e = np.array(e)
    e = np.transpose(e, (0, 2, 1))
    x = np.array(x)
    x = np.transpose(x, (0, 2, 1))

    s = np.squeeze(np.array(s))
    labels = np.squeeze(np.array(labels))
    labels=labels.flatten()
    e = utils.standard_normalize(e)
    x = utils.standard_normalize(x)

    # io.savemat(save_file.format(s), {'eeg': e[:, np.newaxis, :, :],
    #                                  'x': x[:, np.newaxis, :, :], 'y': labels})

    if Ek_po==0:#解决concatenate无法拼接空数组的问题
        X_po=x
        Y_po=labels
        Ek_po=1
    else:
        X_po= np.concatenate((X_po, x), axis=0)
        Y_po= np.concatenate((Y_po, labels), axis=0)


X_cl=X_cl[:, np.newaxis, :, :]
X_po=X_po[:, np.newaxis, :, :]

idx_al=np.arange(0,2240)
idx_cl,_, idx_po, _ = utils.split_data([idx_al, idx_al], split=0.86, shuffle=True)
idx_po,_,idx_test_po,_=utils.split_data([idx_po, idx_po], split=0.5, shuffle=True)
x_train=X_cl[idx_cl]
y_train=Y_cl[idx_cl]
x_poison=X_po[idx_po]
y_poison=Y_po[idx_po]
x_test=X_cl[idx_test_po]
y_test=Y_cl[idx_test_po]
x_test_poison=X_po[idx_test_po]
y_test_poison=Y_po[idx_test_po]

x_train, y_train, x_validation, y_validation = utils.split_data([x_train, y_train], split=0.8, shuffle=True)
save_dir = 'EEG_Data/MI/'
save_file = save_dir + 'data2-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2])

io.savemat(save_file, {'x_train': x_train,'y_train': y_train, 'x_validation':x_validation,'y_validation':y_validation,
                       'x_poison': x_poison,'y_poison':y_poison,'x_test':x_test,'y_test':y_test ,
                       'x_test_poison':x_test_poison,'y_test_poison':y_test_poison})