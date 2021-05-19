import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data
from scipy.io import loadmat
import scipy.io as io

npp_params=[1,5,0.1]
path='EEG_Data/MI/'
data = loadmat(path + 'data2-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2]))
x_train = data['x_train']
y_train = data['y_train']
x_validation = data['x_validation']
y_validation = data['y_validation']
x_poison = data['x_poison']
y_poison = data['y_poison']
x_test= data['x_test']
y_test = data['y_test']
x_test_poison = data['x_test_poison']
y_test_poison = data['y_test_poison']


x_train0=x_train[0][0][0]
x_train1=x_train[0,0,0,:]
mode = pywt.Modes.smooth

def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    w = pywt.Wavelet(w)  # 选取小波函数
    a = data
    ca = []  # 近似分量
    cd = []  # 细节分量
    for i in range(1):###5
        (a, d) = pywt.dwt(a, w, mode)  # 进行5阶离散小波变换
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))  # 重构
    # for i, y in enumerate(rec_a):
    #     if i==2:
    #         return y

    return rec_a

    # for i, coeff in enumerate(cd):
    #     coeff_list = [None, coeff] + [None] * i
    #     if i == 3:
    #         print(len(coeff))
    #         print(len(coeff_list))
    #     rec_d.append(pywt.waverec(coeff_list, w))

    # fig = plt.figure()
    # ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    # ax_main.set_title(title)
    # ax_main.plot(data)
    # ax_main.set_xlim(0, len(data) - 1)
    #
    # for i, y in enumerate(rec_a):
    #     ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
    #     ax.plot(y, 'r')
    #     ax.set_xlim(0, len(y) - 1)
    #     ax.set_ylabel("A%d" % (i + 1))
    #
    # for i, y in enumerate(rec_d):
    #     ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
    #     ax.plot(y, 'g')
    #     ax.set_xlim(0, len(y) - 1)
    #     ax.set_ylabel("D%d" % (i + 1))


# test=np.squeeze(np.array(plot_signal_decomp(x_train0, 'sym5', "DWT: Ecg sample - Symmlets5")))
# test_1=x_train[0][0][0]
for j in range(len(x_train)):
    for k in range(len(x_train[j])):
        for l in range(len(x_train[j][k])):
            x_train[j][k][l]=np.squeeze(np.array(plot_signal_decomp(x_train[j][k][l], 'db2', "pywt")))

for j in range(len(x_validation)):
    for k in range(len(x_validation[j])):
        for l in range(len(x_validation[j][k])):
            x_validation[j][k][l] = np.squeeze(np.array(plot_signal_decomp(x_validation[j][k][l], 'db2', "pywt")))

for j in range(len(x_poison)):
    for k in range(len(x_poison[j])):
        for l in range(len(x_poison[j][k])):
            x_poison[j][k][l] = np.squeeze(np.array(plot_signal_decomp(x_poison[j][k][l], 'db2', "pywt")))

for j in range(len(x_test)):
    for k in range(len(x_test[j])):
        for l in range(len(x_test[j][k])):
            x_test[j][k][l] = np.squeeze(np.array(plot_signal_decomp(x_test[j][k][l], 'db2', "pywt")))


for j in range(len(x_test_poison )):
    for k in range(len(x_test_poison [j])):
        for l in range(len(x_test_poison [j][k])):
            x_test_poison[j][k][l] = np.squeeze(np.array(plot_signal_decomp(x_test_poison[j][k][l], 'db2', "pywt")))


save_dir = 'EEG_Data/MI/'
save_file = save_dir + 'data_pywtdb2-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2])

io.savemat(save_file, {'x_train': x_train,'y_train': y_train, 'x_validation':x_validation,'y_validation':y_validation,
                       'x_poison': x_poison,'y_poison':y_poison,'x_test':x_test,'y_test':y_test ,
                       'x_test_poison':x_test_poison,'y_test_poison':y_test_poison})