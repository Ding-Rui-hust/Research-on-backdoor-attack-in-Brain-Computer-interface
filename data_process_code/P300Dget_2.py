from scipy.signal import butter, filtfilt, resample
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
    B, A = butter(3, np.array(band) / (fs / 2), btype='bandpass')
    return filtfilt(B, A, sig, axis=0)



sample_freq = 2048.0
epoc_window = 1.0 * sample_freq
npp_params=[0.006,   5,   0.1]
data_file = 'EEG_Data/P300/raw/subject{}/session{}'
X_cl=[]
Y_cl=[]
X_po=[]
Y_po=[]
Ek_cl=0
Ek_po=0


for s in tqdm(range(3)):
    x = []
    e = []
    labels = []
    clean=True
    flag = True

    for session in range(4):
        data_names = os.listdir(data_file.format(s + 1, session + 1))
        for data_name in data_names:
            data = io.loadmat(os.path.join(data_file.format(s + 1, session + 1), data_name))
            EEG = data['data']
            EEG = np.transpose(EEG[:-2, :] - np.mean(EEG[[6, 23], :], axis=0), (1, 0))
            events = data['events']
            stimuli = np.squeeze(data['stimuli'])
            target = np.squeeze(data['target'])

            idxs = [((events[i, 3] - events[0, 3]) * 3600.0 + (events[i, 4] - events[0, 4]) * 60.0 + (
                    events[i, 5] - events[0, 5]) + 0.4) * sample_freq for i
                    in range(len(events))]
            y = np.zeros(shape=[len(stimuli)])
            y[np.where(stimuli == target)] = 1#在这里把其余的label全部设为0，符合stimuli的设为1

            if flag:
                labels = y
                flag = False
            else:
                labels = np.concatenate([labels, y])

            if not clean:
                npp = pulse_noise([1, 32, int(0.4 * sample_freq)], freq=npp_params[1], sample_freq=sample_freq,
                                  proportion=npp_params[2])
                amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

                for _, idx in enumerate(idxs):
                    idx = int(idx)
                    EEG[idx:int(idx + 0.4 * sample_freq), :] = np.transpose(npp.squeeze() * amplitude,
                                                                                (1, 0)) + EEG[
                                                                                          idx:int(
                                                                                              idx + 0.4 * sample_freq),
                                                                                          :]
            sig_F = bandpass(EEG, [1.0, 40.0], sample_freq)

            for _, idx in enumerate(idxs):
                idx = int(idx)
                s_EEG = EEG[idx:int(idx + epoc_window), :]
                s_sig = sig_F[idx:int(idx + epoc_window), :]

                s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
                e.append(s_EEG)
                x.append(s_sig)

    e = np.array(e)
    e = np.transpose(e, (0, 2, 1))
    x = np.array(x)
    x = np.transpose(x, (0, 2, 1))
    x = np.clip(x, a_min=-10, a_max=10)

    s = np.squeeze(np.array(s))
    labels = np.squeeze(np.array(labels))
    labels = labels.flatten()
    e = utils.standard_normalize(e)
    x = utils.standard_normalize(x)

    if Ek_cl== 0:  # 解决concatenate无法拼接空数组的问题
        X_cl = x
        Y_cl = labels
        Ek_cl= 1
    else:
        X_cl= np.concatenate((X_cl, x), axis=0)
        Y_cl= np.concatenate((Y_cl, labels), axis=0)

x1 = X_cl[np.where(Y_cl == 0)]  # 消除类别不平衡的问题
x2 = X_cl[np.where(Y_cl == 1)]
sample_num = min(len(x1), len(x2))
idx1, idx2 = utils.shuffle_data(len(x1)), utils.shuffle_data(len(x2))
X_cl= np.concatenate([x1[idx1[:sample_num]], x2[idx2[:sample_num]]], axis=0)
Y_cl= np.concatenate([np.zeros(shape=[sample_num]), np.ones(shape=[sample_num])], axis=0)


for s in tqdm(range(3)):
    x = []
    e = []
    labels = []
    clean=False
    flag = True

    for session in range(4):
        data_names = os.listdir(data_file.format(s + 1, session + 1))
        for data_name in data_names:
            data = io.loadmat(os.path.join(data_file.format(s + 1, session + 1), data_name))
            EEG = data['data']
            EEG = np.transpose(EEG[:-2, :] - np.mean(EEG[[6, 23], :], axis=0), (1, 0))
            events = data['events']
            stimuli = np.squeeze(data['stimuli'])
            target = np.squeeze(data['target'])

            idxs = [((events[i, 3] - events[0, 3]) * 3600.0 + (events[i, 4] - events[0, 4]) * 60.0 + (
                    events[i, 5] - events[0, 5]) + 0.4) * sample_freq for i
                    in range(len(events))]
            y = np.zeros(shape=[len(stimuli)])
            y[np.where(stimuli == target)] = 1#在这里把其余的label全部设为0，符合stimuli的设为1，why?分类精度太低

            if flag:
                labels = y
                flag = False
            else:
                labels = np.concatenate([labels, y])

            if not clean:
                npp = pulse_noise([1, 32, int(0.4 * sample_freq)], freq=npp_params[1], sample_freq=sample_freq,
                                  proportion=npp_params[2])
                amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

                for _, idx in enumerate(idxs):
                    idx = int(idx)
                    EEG[idx:int(idx + 0.4 * sample_freq), :] = np.transpose(npp.squeeze() * amplitude,
                                                                                (1, 0)) + EEG[
                                                                                          idx:int(
                                                                                              idx + 0.4 * sample_freq),
                                                                                          :]
            sig_F = bandpass(EEG, [1.0, 40.0], sample_freq)

            for _, idx in enumerate(idxs):
                idx = int(idx)
                s_EEG = EEG[idx:int(idx + epoc_window), :]
                s_sig = sig_F[idx:int(idx + epoc_window), :]

                s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
                e.append(s_EEG)
                x.append(s_sig)

    e = np.array(e)
    e = np.transpose(e, (0, 2, 1))
    x = np.array(x)
    x = np.transpose(x, (0, 2, 1))
    x = np.clip(x, a_min=-10, a_max=10)

    s = np.squeeze(np.array(s))
    labels = np.squeeze(np.array(labels))
    labels = labels.flatten()
    e = utils.standard_normalize(e)
    x = utils.standard_normalize(x)

    if Ek_po== 0:  # 解决concatenate无法拼接空数组的问题
        X_po = x
        Y_po = labels
        Ek_po= 1
    else:
        X_po= np.concatenate((X_po, x), axis=0)
        Y_po= np.concatenate((Y_po, labels), axis=0)

x1 = X_po[np.where(Y_po == 0)]  # 消除类别不平衡的问题
x2 = X_po[np.where(Y_po == 1)]
sample_num = min(len(x1), len(x2))
#idx1, idx2 = utils.shuffle_data(len(x1)), utils.shuffle_data(len(x2))
X_po= np.concatenate([x1[idx1[:sample_num]], x2[idx2[:sample_num]]], axis=0)
Y_po= np.concatenate([np.zeros(shape=[sample_num]), np.ones(shape=[sample_num])], axis=0)


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
x_test_poison=X_po[idx_test_po]
y_test_poison=Y_po[idx_test_po]

x_train, y_train, x_validation, y_validation = utils.split_data([x_train, y_train], split=0.8, shuffle=True)
save_dir = 'EEG_Data/P300/'
save_file = save_dir + 'data2-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2])

io.savemat(save_file, {'x_train': x_train,'y_train': y_train, 'x_validation':x_validation,'y_validation':y_validation,
                       'x_poison': x_poison,'y_poison':y_poison,'x_test':x_test,'y_test':y_test ,
                       'x_test_poison':x_test_poison,'y_test_poison':y_test_poison})