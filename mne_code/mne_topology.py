import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


#设置通道名
biosemi_montage = mne.channels.make_standard_montage('biosemi64')
#生成数据
data = np.random.randn(64,1)


#创建info对象
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=128.,
                            ch_types='eeg')
#创建evokeds对象
evoked = mne.EvokedArray(data, info)
#evokeds设置通道
evoked.set_montage(biosemi_montage)
#画图
mne.viz.plot_topomap(evoked.data[:, 0], evoked.info,show=False)
plt.show()