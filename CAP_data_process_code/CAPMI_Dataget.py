from scipy.signal import butter, lfilter, resample,iirnotch
from tqdm import tqdm
import scipy.io as io
import numpy as np
import pandas as pd
import lib.utils as utils
import sys



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
epoc_window =2* sample_freq#可以变

subjects = ['0003','0004','0005','0006','0007','0008','0009','0010','0011','0012','0013','0014','0015','0016','0017',
'0018','0019','0020','0021','0022','0023','0024','0025','0026','0027','0028','0029','0030','0031','0032','0033','0034',
'0035','0036']
npp_params=[0.3,   5,   0.1]
data_file = 'EEG_Data/MI_DR2/raw/data_{}_raw.csv'


y=np.squeeze(np.array([1,1,0,1,0,1,1,0,0,1,
                       0,0,1,0,1,0,1,0,0,0,
                       1,0,1,0,1,1,0,1,1,1]))
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
    #idxFeedBack2 = np.where(Trigger == 2800)[0]

    sig_F = bandpass(EEG, [8.0, 30.0], sample_freq)
    sig_F= notch_filtering(sig_F,sample_freq, 50 , 30)


    for i, idx in enumerate(idxFeedBack):
        idx = int(idx)
        ################基线修正
        # idx2 = idxFeedBack2[i]  # 基线修正
        # #a = sig_F[idx2:int(idx2 + 100), :]
        # mul_mean = np.mean(sig_F[idx2:int(idx2 + 200), :], axis=0)
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
    # # 创建PCA的计算模型_不完整
    # ica = UnsupervisedSpatialFilter(FastICA(16), average=False)
    # # 进行PCA处理
    # x = ica.fit_transform(x)
    s = np.squeeze(np.array(s))
    y = np.squeeze(np.array(y))
    x = utils.standard_normalize(x)

    if Ek_cl==0:#解决concatenate无法拼接空数组的问题
        X_cl=x
        Y_cl=y
        Ek_cl=1
    else:
        X_cl= np.concatenate((X_cl, x), axis=0)
        Y_cl= np.concatenate((Y_cl, y), axis=0)


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
        npp = pulse_noise([1, 16, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                          proportion=npp_params[2])
        amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

        for _, idx in enumerate(idxFeedBack):
            idx = int(idx)
            EEG[idx:int(idx + epoc_window), :] = np.transpose(npp.squeeze() * amplitude,
                                                                  (1, 0)) + EEG[idx:int(idx + epoc_window), :]

    sig_F = bandpass(EEG, [8.0, 30.0], sample_freq)
    sig_F = notch_filtering(sig_F, sample_freq, 50, 30)#f=50,Q=30
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

    if Ek_po==0:#解决concatenate无法拼接空数组的问题
        X_po=x
        Y_po=y
        Ek_po=1
    else:
        X_po= np.concatenate((X_po, x), axis=0)
        Y_po= np.concatenate((Y_po, y), axis=0)


ampl_al=np.std(X_cl, axis=(1,2))
ind=np.argsort(ampl_al)
ind_vaild=ind[30:-30]
X_cl2=X_cl[ind_vaild]
X_po2=X_po[ind_vaild]
Y_cl2=Y_cl[ind_vaild]
Y_po2=Y_po[ind_vaild]

X_cl2=X_cl2[:, np.newaxis, :, :]
X_po2=X_po2[:, np.newaxis, :, :]

idx_al2=np.arange(0,960)###########
idx_cl2,_, idx_po2, _ = utils.split_data([idx_al2, idx_al2], split=0.86, shuffle=True)
idx_po2,_,idx_test_po2,_=utils.split_data([idx_po2, idx_po2], split=0.5, shuffle=True)
x_train=X_cl2[idx_cl2]
y_train=Y_cl2[idx_cl2]
x_poison=X_po2[idx_po2]
y_poison=Y_po2[idx_po2]
x_test=X_cl2[idx_test_po2]
y_test=Y_cl2[idx_test_po2]
x_test_poison=X_po2[idx_test_po2]
y_test_poison=Y_po2[idx_test_po2]

x_train, y_train, x_validation, y_validation = utils.split_data([x_train, y_train], split=0.8, shuffle=True)
save_dir = 'EEG_Data/MI_DR2/'
save_file = save_dir + 'data2-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2])

# io.savemat(save_file, {'x_train': x_train,'y_train': y_train, 'x_validation':x_validation,'y_validation':y_validation,
#                        'x_poison': x_poison,'y_poison':y_poison,'x_test':x_test,'y_test':y_test ,
#                        'x_test_poison':x_test_poison,'y_test_poison':y_test_poison})