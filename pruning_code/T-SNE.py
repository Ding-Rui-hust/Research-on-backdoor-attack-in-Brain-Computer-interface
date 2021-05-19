import h5py
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model,Model
from scipy.io import loadmat
import tensorflow.compat.v1.keras.backend as K
K.set_image_data_format('channels_first')
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# 先定义我们读取keras模型权重的函数
def print_keras_wegiths(weight_file_path):
    # 读取weights h5文件返回File类
    f = h5py.File(weight_file_path)
    try:
        # 读取各层的名称以及包含层信息的Group类
        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))
    finally:
        f.close()


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

y_train=np.squeeze(y_train)
y_validation=np.squeeze(y_validation)
y_poison=np.squeeze(y_poison)
y_test=np.squeeze(y_test)
y_test_poison=np.squeeze(y_test_poison)

#model=load_model(filepath='model_orginal_poison_before{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
model0 = load_model(filepath='model_orginal_poison_beforeMI{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
#model = load_model(filepath='model_orginal_gauss0.1{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
#model = load_model(filepath='model_orginal_poison_MIDRpruning-{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
#model = load_model(filepath='model_orginal_poison_MIDRgausspruning6-{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))

print(model0.summary())



# 我们再来看一下模型的各层，选择某一层的特征输出作为降维数据


print("Using loaded model to predict...")

model = Model(inputs=model0.input, outputs=model0.get_layer('flatten').output)

# 读取要做tsne降维的数据，shape=(1000,28,28,1)


# 用模型得到预测结果，进而得到降维后的结果
predict = model.predict(x_train)
print(predict.shape)
tsne = TSNE(n_components=2, learning_rate=300, init='pca', random_state=0)
X_tsne_0 = tsne.fit_transform(predict)

# 利用数据的label和降维的数据画图
for j in range(predict.shape[0]):#predict.shape[0]
    #if X_tsne_0[j, 0] <= -5 and X_tsne_0[j, 0] >= -7 and X_tsne_0[j, 1] <= 45 and X_tsne_0[j, 1] >= -33:
    if y_train[j] == 0:  # label=75的类
        plt.scatter(X_tsne_0[j, 0], X_tsne_0[j, 1], marker='x', c='b')
    elif y_train[j] == 1:  # label=10的类
        plt.scatter(X_tsne_0[j, 0], X_tsne_0[j, 1], marker='x', c='r')


# plt.xticks(np.linspace(-7, -5.5, 10))
# plt.yticks(np.linspace(-0.5, 1.0, 10))
plt.show()
