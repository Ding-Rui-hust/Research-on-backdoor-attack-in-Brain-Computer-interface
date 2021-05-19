import mne
import numpy as np
import scipy.io
from scipy.io import loadmat
import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg') #You can use this backend if needed
# plt.ion()#使plot可互动


npp_params=[0.3,5,0.1]
path='EEG_Data/MI_DR02/'
data = loadmat(path + 'data-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2]))
#data = loadmat(path + 'dataPOI0.1.mat')
x_train = data['x_train']
x_train=np.squeeze(x_train)
x_train=x_train*0.000033
x_train0=x_train[0]
for i in range(1,len(x_train)):
    x_train0=np.concatenate((x_train0,x_train[i]),axis=1)

# samplesfile = scipy.io.loadmat('mateegdata.mat') #文件读入字典
# samples = samplesfile['eegdata'] #提取字典中的numpy数组

#ch_names = ['1', '2','3','4','5','6', '7','8','9','10','11', '12','13','14','15'] #通道名称
ch_names = ['P3','C3','F3','Fz','F4','C4','P4','Cz','T3','T5','O1','O2','F7','F8','T6','T4']
ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg','eeg', 'eeg','eeg', 'eeg']
sfreq = 128 #采样率

info = mne.create_info(ch_names=ch_names,ch_types=ch_types,sfreq=sfreq) #创建信号的信息
info.set_montage('standard_1020')
raw = mne.io.RawArray(x_train0, info)

#raw.plot(block=True)
print('数据集的形状为：',raw.get_data().shape)
print('通道数为：',raw.info.get('nchan'))

#raw.plot_projs_topomap()
#plt.show()


#########################绘制某一通道频率响应
# raw.plot_psd()
# plt.show()
###########################

