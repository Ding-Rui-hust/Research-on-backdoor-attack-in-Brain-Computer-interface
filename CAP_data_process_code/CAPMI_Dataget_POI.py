from scipy.signal import butter, lfilter, resample,iirnotch
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

def notch_filtering(wav, fs, w0, Q):#陷波滤波器
    """ Apply a notch (band-stop) filter to the audio signal.

    Args:
        wav: Waveform.
        fs: Sampling frequency of the waveform.
        w0: See scipy.signal.iirnotch.
        Q: See scipy.signal.iirnotch.

    Returns:
        wav: Filtered waveform.
    """
    b, a = iirnotch(2 * w0/fs, Q)
    wav = lfilter(b, a, wav)
    return wav

sample_freq = 300.0
epoc_window = 2 * sample_freq#可以变

# subjects = ['0001','0002','0003','0004','0005','0006','0007','0008','0009','0010','0011','0012','0013','0014','0015','0016','0017',
# '0018','0019','0023','0021','0022','0023','0024','0025','0026','0027','0028']

subjects = ['0037','0038']
npp_params=[0.2,   5,   0.1]
data_file = 'EEG_Data/MI_DR02/raw/data_{}_raw.csv'

y1=np.squeeze(np.array([1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
                       1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1]))
y2=np.squeeze(np.array([1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
                       1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1]))

# y1=np.squeeze(np.array([0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
#                        0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0]))
# y2=np.squeeze(np.array([0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
#                        0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0]))



X_cl=[]
Y_cl=[]
X_po=[]
Y_po=[]
Ek_cl=0
Ek_po=0
q1=0
q2=0

for index in tqdm(range(len(subjects))):
    x = []
    e = []
    s = []
    clean=True
    q1+=1
    file_name = data_file.format(subjects[index])
    sig = np.array(pd.read_csv(file_name).values)########读取CSV文件

    EEG = sig[:, 1:-2]#取每行从第二个到倒数第三个（去掉最后两个）
    Trigger = sig[:, -1]#取每行的最后一位

    idxFeedBack = np.where(Trigger == 1)[0]
    idxFeedBack2 = np.where(Trigger == 1700)[0]

    if not clean:
        npp = pulse_noise([1, 16, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                          proportion=npp_params[2])
        amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

        for _, idx in enumerate(idxFeedBack):
            idx = int(idx)
            EEG[idx:int(idx + epoc_window), :] = np.transpose(npp.squeeze() * amplitude,
                                                                  (1, 0)) + EEG[idx:int(idx + epoc_window), :]

    sig_F = bandpass(EEG, [8.0, 30.0], sample_freq)
    sig_F= notch_filtering(sig_F, sample_freq, 50, 30)


    for i, idx in enumerate(idxFeedBack):
        idx = int(idx)
        ################基线修正
        # idx2 = idxFeedBack2[i]  # 基线修正
        # #a = sig_F[idx2:int(idx2 + 100), :]
        # mul_mean = np.mean(sig_F[idx2:int(idx2 + 100), :], axis=0)
        # s_sig = sig_F[idx:int(idx + epoc_window), :]
        # b = np.zeros((int(epoc_window), 16))
        # for j in range(16):
        #     b[:, j] = mul_mean[j]
        # s_sig = s_sig - b
        ###############################
        s_sig = sig_F[idx:int(idx + epoc_window), :]
        s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
        x.append(s_sig)
        s.append(idx)

    x = np.array(x)
    x = np.transpose(x, (0, 2, 1))
    # # 创建PCA的计算模型
    # ica = UnsupervisedSpatialFilter(FastICA(16), average=False)
    # # 进行PCA处理
    # x = ica.fit_transform(x)


    s = np.squeeze(np.array(s))
    if q1%2==1:
        y = np.squeeze(np.array(y1))
    else:
        y=  np.squeeze(np.array(y2))
    x = utils.standard_normalize(x)


    if Ek_cl==0:#解决concatenate无法拼接空数组的问题
        X_cl=x
        Y_cl=y
        Ek_cl=1
    else:
        X_cl= np.concatenate((X_cl, x), axis=0)
        Y_cl= np.concatenate((Y_cl, y), axis=0)


ampl_al=np.std(X_cl, axis=(1,2))
ind=np.argsort(ampl_al)
ind_vaild = ind[5:-5]
X_cl2=X_cl[ind_vaild]
Y_cl2=Y_cl[ind_vaild]
X_cl2=X_cl2[:, np.newaxis, :, :]
save_dir = 'EEG_Data/MI_DR02/'
save_file = save_dir + 'dataPOI0.05.mat'

io.savemat(save_file, {'x_POI': X_cl2,'y_POI': Y_cl2})