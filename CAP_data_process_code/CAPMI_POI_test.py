from tensorflow.keras.models import load_model
from scipy.io import loadmat
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import models,models2
import tensorflow.compat.v1.keras.backend as K

#npp_params=[0.05,   5,   0.1]
#NPP=0.1
path='EEG_Data/MI_DR02/'
data = loadmat(path + 'dataPOI2_0.2.mat')
x_POI = data['x_POI']
y_POI = data['y_POI']
y_POI=np.squeeze(y_POI)


model=load_model(filepath='model_CNNPOI0.2.h5')


#ASR检测，与之前的检测不同，不需要也无法去除分类有误的数据，直接计算原来标签为0且网络预测后变为1的数据 占标签为0数据的比例
y_pred = np.argmax(model.predict(x_POI), axis=1)
print(y_pred)
idx = np.where(y_POI == 0)  # 看哪些原来不是1，但是又被判为1了
x_POI, y_POI = x_POI[idx], y_POI[idx]  # 原来判断为1的地方全都去掉了
p_pred = np.argmax(model.predict(x_POI), axis=1)  # 在均为0.5时卡在这里
poison_s_rate = 1 - np.sum(p_pred == y_POI).astype(np.float32) / len(p_pred)  # 用1减去对poisontest预测与test预测一样的样本占总样本的比例
print('poison attack success rate: {}'.format(poison_s_rate))


