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

SNR=0.1
npp_params = [0.2, 5, 0.1]
path='EEG_Data/ERN/'
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

model=load_model(filepath='model_poison_CNNERN_{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
#model=load_model(filepath='model_orginal_poison_ERNcleanretrain{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))

print(model.summary())
y_pred = np.argmax(model.predict(x_test), axis=1)
acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)
print('acc-{} '.format(acc))

# 在剪枝前测试网络情况
idx = y_pred == y_test
x_t, y_t = x_test_poison[idx], y_test[idx]
idx = np.where(y_t == 0)
x_t, y_t = x_t[idx], y_t[idx]
p_pred = np.argmax(model.predict(x_t), axis=1)
poison_s_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)
print('poison attack success rate: {}'.format(poison_s_rate))
# #y_pred = np.argmax(model.predict(x_train), axis=1)
# #for i in range(25):
names = [layer.name for layer in model.layers]
print(names, len(names))
# layer_model = Model(inputs=model.input, outputs=model.get_layer('flatten').output)#原计划对flatten层剪枝
# feature=layer_model.predict(x_train)
# feature_mean=np.mean(abs(feature),axis=0)#是否要abs
# ind=np.argsort(np.abs(feature_mean),axis=None)
# print(ind)
pruning_norm=[50,100,200]#DeepCNN后三个卷积层filter的个数
pruning_idx=[30,60,150]#要减去的filter的个数
##########################################################
layer_model = Model(inputs=model.input, outputs=model.get_layer('max_pooling2d_1').output)#以pooling层的输出作为判断神经元（filter）是否休眠的依据
feature=layer_model.predict(x_train)#观察在输入训练集时哪些filter休眠
#a=abs(feature)
feature_mean=np.mean(np.abs(feature),axis=(0,2,3))#是否要abs---如何判断是否休眠？（依据fine_pruning文章确定的判断值：对输出取绝对值后再取平均值）
ind=np.argsort(feature_mean)
print(ind)

sparse_model = tf.keras.models.clone_model(model)
sparse_model.set_weights(model.get_weights())
weights = sparse_model.get_weights()
#weight_3=sparse_model.get_layer('conv2d_4').get_weights()
count = 0
for i in range(pruning_idx[0]):#剪枝
    channel = ind[i]
    weights[8][:, :, :, channel] = 0.#w置零：与这一池化层对应的卷积层参数在weight的8,9，需要匹配卷积神经网络的结构（summary）和权重(get_weights)
    weights[9][channel] = 0.         #b置零
    count += 1



###########################################################
layer_model = Model(inputs=model.input, outputs=model.get_layer('max_pooling2d_2').output)
feature=layer_model.predict(x_train)
#a=abs(feature)
feature_mean=np.mean(np.abs(feature),axis=(0,2,3))
ind=np.argsort(feature_mean)
print(ind)


#weight_3=sparse_model.get_layer('conv2d_4').get_weights()
count = 0
for i in range(pruning_idx[1]):
    channel = ind[i]
    weights[14][:, :, :, channel] = 0.
    weights[15][channel] = 0.
    count += 1


#################################DeepCNN
layer_model = Model(inputs=model.input, outputs=model.get_layer('max_pooling2d_3').output)
feature=layer_model.predict(x_train)
#a=abs(feature)
feature_mean=np.mean(np.abs(feature),axis=(0,2,3))
ind=np.argsort(feature_mean)
print(ind)

# sparse_model = tf.keras.models.clone_model(model)
# sparse_model.set_weights(model.get_weights())
# weights = sparse_model.get_weights()
#weight_4=sparse_model.get_layer('conv2d_4').get_weights()
count = 0
for i in range(pruning_idx[2]):
    channel = ind[i]
    weights[20][:, :, :, channel] = 0.
    weights[21][channel] = 0.
    count += 1
print(count)
sparse_model.set_weights(weights)
###########################################




y_pred = np.argmax(sparse_model.predict(x_test), axis=1)
acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)  #
print('acc-{} '.format( acc))

# 判断剪枝后的ACC与ASR
idx = y_pred == y_test
x_t, y_t = x_test_poison[idx], y_test[idx]
idx = np.where(y_t == 0)
x_t, y_t = x_t[idx], y_t[idx]
p_pred = np.argmax(sparse_model.predict(x_t), axis=1)
poison_s_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)
print('poison attack success rate: {}'.format(poison_s_rate))

#sparse_model.save(filepath='model_CNNERN_pruning_{}_{}_{}_{}.h5'.format(npp_params[0], pruning_idx[0], pruning_idx[1],pruning_idx[2]))


