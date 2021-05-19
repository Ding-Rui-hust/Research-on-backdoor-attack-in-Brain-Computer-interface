import numpy as np
from scipy.io import loadmat
import scipy.io as io
import lib.utils as utils
npp_params=[1.5,5,0.1]
path='EEG_Data/MI/'
data = loadmat(path + 'data2-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2]))
signal = data['x_train']
y_gusspoi = np.squeeze(data['y_train'])
signal , y_gusspoi ,_  , _ = utils.split_data([signal, y_gusspoi], split=0.2, shuffle=True)
#对信号添加指定信噪比的高斯信号
signal_noise=np.zeros(signal.shape)
SNR = 0.1
for i in range(len(signal)):
    noise = np.random.randn(signal.shape[1],signal.shape[2],signal.shape[3]) 	#产生N(0,1)噪声数据
    noise = noise-np.mean(noise) 	#均值为0
    a=signal[i].size
    signal_power = np.linalg.norm( signal[i] - signal[i].mean() )**2 / signal[i].size	#此处是信号的std**2
    noise_variance = signal_power/np.power(10,(SNR/10))         #此处是噪声的std**2
    noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    ##此处是噪声的std**2
    signal_noise[i] = noise + signal[i]

Ps = ( np.linalg.norm(signal - signal.mean()) )**2          #signal power
Pn = ( np.linalg.norm(signal - signal_noise) )**2          #noise power
snr = 10*np.log10(Ps/Pn)
print(snr)



# save_dir = 'EEG_Data/MI/'
# save_file = save_dir + 'gausspoi-{}-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2],SNR)
# io.savemat(save_file, {'x_gausspoi':signal_noise,'y_gausspoi': y_gusspoi,'x_signal':signal,'y_signal':y_gusspoi})