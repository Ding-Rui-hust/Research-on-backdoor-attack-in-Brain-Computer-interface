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
data_evoked = x_test.mean(0)
# epochs的数量
nave = x_test.shape[0]
comment = "MI_DR"

# 设置事件开始前时间为-0.1s
tmin = 0
evoked_array = mne.EvokedArray(data_evoked, info, tmin,comment=comment, nave=nave)
print(evoked_array)
# _ = evoked_array.plot(time_unit='s')

evoked_array.plot_image(exclude=[], time_unit='s')


# evoked_array.plot(spatial_colors=True, gfp=True, picks='eeg')


#evoked_array.plot_topomap(time_unit='s')#画出脑部分布图
##################################################################topomap
# evoked_array.plot_topomap(times='peaks', ch_type='eeg', time_unit='s',title='MI_DR',colorbar=False)
# plt.tight_layout()
# plt.show()
##################################################################

############################################################画出脑部组合图
# ts_args = dict(gfp=True, time_unit='s')
# topomap_args = dict(sensors=False, time_unit='s')
# evoked_array.plot_joint(title='MI_DR', times=[0.2, 0.6, 1.0, 1.4, 1.8],ts_args=ts_args, topomap_args=topomap_args)

##########################################################画电极位置

# evoked_array.plot_sensors(title='MI_DR',show_names=True)
# plt.show()
#############################################

# evoked_array.plot(exclude=[], time_unit='s')
# plt.show()
###################################


