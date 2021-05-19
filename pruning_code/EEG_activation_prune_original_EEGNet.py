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
K.set_image_data_format('channels_first')
from tensorflow.keras.callbacks import EarlyStopping

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#SNR=0.004
npp_params = [0.3, 5, 0.1]
path='EEG_Data/MI_DR02/'
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
y_train=np.squeeze(y_train)
y_validation=np.squeeze(y_validation)
y_poison=np.squeeze(y_poison)
y_test=np.squeeze(y_test)
y_test_poison=np.squeeze(y_test_poison)

# model=load_model(filepath='model_orginal_poison_beforeDRMI{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
model=load_model(filepath='model_poison_MIDR_{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
print(model.summary())
y_pred = np.argmax(model.predict(x_test), axis=1)
acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)  # 这里人工设计的acc与之前自动生成的val_acc的区别是什么？
print('acc-{} '.format(acc))

# poison performance
idx = y_pred == y_test  # 判断前是否等于后，输出true or false(这里是在干嘛)#有的时候一个也没对，idx为空
x_t, y_t = x_test_poison[idx], y_test[idx]  # 判断为错误的地方全部去掉了，原来160，现在acc=0.6675，x_t只有107的长度
idx = np.where(y_t == 0)  # 看哪些原来不是1，但是又被判为1了
x_t, y_t = x_t[idx], y_t[idx]  # 原来判断为1的地方全都去掉了
p_pred = np.argmax(model.predict(x_t), axis=1)  # 在均为0.5时卡在这里
poison_s_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)  # 用1减去对poisontest预测与test预测一样的样本占总样本的比例
print('poison attack success rate: {}'.format(poison_s_rate))
# #y_pred = np.argmax(model.predict(x_train), axis=1)
# #for i in range(25):
names = [layer.name for layer in model.layers]
print(names, len(names))
# layer_model = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
# feature=layer_model.predict(x_train)
# feature_mean=np.mean(abs(feature),axis=0)#是否要abs
# ind=np.argsort(np.abs(feature_mean),axis=None)
# print(ind)




#################################
layer_model = Model(inputs=model.input, outputs=model.get_layer('average_pooling2d_1').output)
feature=layer_model.predict(x_train)
#a=abs(feature)
feature_mean=np.mean(np.abs(feature),axis=(0,2,3))#是否要abs---如何判断是否休眠？-用于EEGNet非常适合，且若剪后一半对ASR没有影响
ind=np.argsort(feature_mean)
print(ind)

sparse_model = tf.keras.models.clone_model(model)
sparse_model.set_weights(model.get_weights())
weights = sparse_model.get_weights()
#weight_4=sparse_model.get_layer('separable_conv2d').get_weights()
count = 0
for i in range(6):#删减6/8
    channel = ind[i]
    weights[11][:, :, :, channel] = 0.#SeparableConv2D这一层的参数在weight11的位置
    #weights[21][channel] = 0.
    count += 1
print(count)
sparse_model.set_weights(weights)
###########################################

y_pred = np.argmax(sparse_model.predict(x_test), axis=1)
acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)
print('acc-{} '.format( acc))

# 判断剪枝后的ACC与ASR
idx = y_pred == y_test
x_t, y_t = x_test_poison[idx], y_test[idx]
idx = np.where(y_t == 0)
x_t, y_t = x_t[idx], y_t[idx]
p_pred = np.argmax(sparse_model.predict(x_t), axis=1)
poison_s_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)
print('poison attack success rate: {}'.format(poison_s_rate))

#sparse_model.save(filepath='model_poison_MIDR_pruning6_{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))