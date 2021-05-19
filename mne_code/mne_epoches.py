import mne
import numpy as np
import scipy.io
from scipy.io import loadmat
import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg') #You can use this backend if needed
# plt.ion()#使plot可互动

npp_params=[0.3,5,0.1]
path='EEG_Data/MI_DR02/'
data = loadmat(path + 'data2-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2]))
#data = loadmat(path + 'dataPOI0.1.mat')

x_test = data['x_test']
x_test=np.squeeze(x_test)
x_test=x_test*0.0001#调整幅值
y_test = data['y_test']
y_test = np.squeeze(y_test)

sfreq=128

a = np.zeros(len(y_test))
for i in range(len(y_test)):
    a[i] = i
b=np.zeros(len(y_test))
a=a.astype(np.int)
b=b.astype(np.int)
y_test=y_test.astype(np.int)
events=np.vstack((a,b,y_test)).T#组合数组
#events=b.append(y_train)
#events=np.concatenate(b,y_train)

#ch_names = ['1', '2','3','4','5','6', '7','8','9','10','11', '12','13','14','15','16']
ch_names = ['P3','C3','F3','Fz','F4','C4','P4','Cz','T3','T5','O1','O2','F7','F8','T6','T4']
ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg','eeg']
#ch_types = ['grad', 'grad', 'grad', 'grad', 'grad', 'grad','grad', 'grad', 'grad','grad', 'grad', 'grad','grad', 'grad', 'grad']
info = mne.create_info(ch_names=ch_names,ch_types=ch_types,sfreq=sfreq)
info.set_montage('standard_1020')
event_id = dict(left=0, right=1)

# 设置事件开始前时间为-0.1s
tmin = 0
"""
利用mne.EpochsArray创建epochs对象
"""
epochs = mne.EpochsArray(x_test, info, events, tmin, event_id)
print(epochs)
print(epochs.event_id)
#epochs.plot(block=True)
#plt.show()
###################################################
#epochs.save('epoches_DR.fif')
###################################################
#epochs.plot(block=True)
#######################################################
#_ = epochs['left'].average().plot(time_unit='s')
#plt.show()

######################################################
# epochs.plot_image(14,
#                   cmap='interactive',
#                   sigma=1.,
#                   vmin=-4, vmax=4)
##########################################################

#epochs.plot_image(combine='gfp', sigma=2., cmap="YlGnBu_r")

######################################################

# epochs.plot_topo_image(vmin=-8,
#                        vmax=8,
#                        title='ERF images',
#                        sigma=2.,
#                        fig_facecolor='w',
#                        font_color='k')
#####################################################
epochs.plot_psd(picks='eeg')
######################################################
