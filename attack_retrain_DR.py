import lib.utils as utils
from lib.load_data import load
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K #暂时只能这样，换成别的版本会报错
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
from scipy.io import loadmat
import numpy as np
import argparse
import os
import models,models2,models3
import sys
sys.path.append('..')
import scipy.io as io
K.set_image_data_format('channels_first')
#数据格式为channels_first，默认的为channel_last，这样使用的原因是在EEGNet原文献中如此使用
#这样必须使用GPU来进行运算，并且配置有误易报错

parser = argparse.ArgumentParser()#参数解释器
parser.add_argument('--gpu_n', type=int, default=0, help='name of GPU')#使用者的GPU名称
parser.add_argument('--data', type=str, default='MI', help='name of data, ERN or MI or P300 or MI_DR')
parser.add_argument('--model', type=str, default='EEGNet', help='name of model, EEGNet or DeepCNN')
parser.add_argument('--a', type=float, default=1.5, help='NPP amplitude')#NPP幅值与标准差的比，由经验和实际效果得出
parser.add_argument('--f', type=int, default=5, help='NPP freq')#NPP信号频率，全文设定为5Hz
parser.add_argument('--p', type=float, default=0.1, help='NPP proportion')#NPP信号占空比
parser.add_argument('--pr', type=float, default=0.1, help='poison_rate')#嵌入后门的信号占总信号数的比例
parser.add_argument('--baseline', type=bool, default=False, help='is baseline')#baseline为True代表在输入中不添加被NPP信号污染的数据
POI=True#在实际攻击中的NPP信号添加

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_n)
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

random_seed = None
#subject_numbers = {'ERN': 16, 'MI': 14, 'P300': 8}
data_name = opt.data  # 'ERN' or 'MI' or 'P300'
model_used = opt.model  # 'EEGNet' or 'DeepCNN'
npp_params = [opt.a, opt.f, opt.p]
batch_size = 16 #batch_size在MIDR数据集上为8，在MI、P300、ERN数据集上为32
epoches = 1600 #设置最多1600次，会提前停止，一般不会超过100个epoch
repeat = 10 #每个数据重复10次取平均值，箱型图的数据也是依据计算10次的数据绘制
poison_rate = opt.pr
baseline = opt.baseline

###################################读取数据
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

# path='EEG_Data/MI_DR02/'#
# data = loadmat(path + 'dataPOI0.2.mat')
# x_POI = data['x_POI']
# y_POI = data['y_POI']
# y_POI=np.squeeze(y_POI)



y_poison = np.ones(shape = y_poison.shape)  # 这里是决定是否改变加poison的数据的标签


#x_train, y_train, x_validation, y_validation = utils.split_data([x_validation, y_validation ], split=0.8, shuffle=True)#减少x_train的数量，避免retrain时干净数据太多
x_train, y_train, _, _= utils.split_data([x_train, y_train], split=0.2, shuffle=True)
x_validation, y_validation ,  _, _= utils.split_data([x_validation, y_validation], split=0.5, shuffle=True)

if not baseline:#若不是baseline，加上污染（后门）数据
    x_train = np.concatenate((x_train, x_poison), axis=0)
    y_train = np.concatenate((y_train, y_poison), axis=0)

# if  POI:
#     x_train = np.concatenate((x_train, x_POI), axis=0)
#     y_train = np.concatenate((y_train, y_POI), axis=0)


data_size = y_train.shape[0]#打乱数据顺序，重新排序
shuffle_index = utils.shuffle_data(data_size)
x_train = x_train[shuffle_index]
y_train = y_train[shuffle_index]

print(x_train.shape)
nb_classes = len(np.unique(y_train))  # np.unique是去除重复数字然后排序输出，这样输出它的长度表明总共的分类数-可尝试增加分类数看结果的变化
samples = x_train.shape[3]
channels = x_train.shape[2]

racc = []
rbca = []
rasr = []
pruning_idx=[30,60,180]#CNN剪枝的参数
for i in range(repeat):

    # Build Model
    if model_used == 'EEGNet':
        model = models.EEGNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
    elif model_used == 'DeepCNN':
        model = models.DeepConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
    else:
        raise Exception('No such model:{}'.format(model_used))

    #######读取之前存储的模型
    model=load_model(filepath='model_poison_MI_{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
    #model = load_model(filepath='model_poison_ERN_pruning6_{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
    #model = load_model(filepath='model_orginal_retrain{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
    #model = load_model(filepath='model_CNNERN_pruning_{}_{}_{}_{}.h5'.format(npp_params[0], pruning_idx[0], pruning_idx[1],pruning_idx[2]))
    #model = load_model(filepath='model_orginal_poison_CNNP300pruning-{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc', ])
    early_stop = EarlyStopping(monitor='val_acc', mode='max', patience=20)  # 容忍多少个epoch里没有明显提升就提前结束

    # #####训练模型
    his = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_data=(x_validation, y_validation),
        shuffle=True,
        epochs=epoches,
        callbacks=[early_stop],
    )



    #### 测试模型
    y_pred = np.argmax(model.predict(x_test), axis=1)
    bca = utils.bca(y_test, y_pred)
    acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)  #计算分类准确度
    print('{}_{}: acc-{} bca-{}'.format(data_name, model_used, acc, bca))


    # 计算攻击成功率（在分类正确的基础上）
    idx = y_pred == y_test  # 判断前是否等于后，输出true or false
    x_t, y_t = x_test_poison[idx], y_test[idx]#判断为错误的地方全部去掉
    idx = np.where(y_t == 0)#看哪些原来不是1，但是又被判为1了
    x_t, y_t = x_t[idx], y_t[idx]#原来判断为1的地方全都去掉了
    p_pred = np.argmax(model.predict(x_t), axis=1)  # 在均为0.5时卡在这里
    poison_s_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)#用1减去对poisontest预测与test预测一样的样本占总样本的比例
    print('poison attack success rate: {}'.format(poison_s_rate))
    K.clear_session()

    racc.append(acc)
    rbca.append(bca)
    rasr.append(poison_s_rate)



print('racc:', racc)
print('rbca:', rbca)
print('rpoison_rate:', rasr)
print('Mean RCA={}, mean BCA={}, mean ASR={}'.format(np.mean(racc), np.mean(rbca), np.mean(rasr)))


#################存储计算结果以画箱型图npz
# results_dir = 'results/{}_{}'.format(data_name,model_used)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)
#
# if not baseline:
#     np.savez(results_dir + '/npp_{}_{}_{}.npz'.format(npp_params[0], npp_params[1], npp_params[2]), racc=racc,
#              rbca=rbca, rasr=rasr)
# else:
#     np.savez(results_dir + '/baseline_{}_{}_{}.npz'.format(npp_params[0], npp_params[1], npp_params[2]), racc=racc,
#              rbca=rbca, rasr=rasr)



#将计算完成的模型进行存储
#model.save(filepath='model_pruning_retrain{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))  # 存模型，放到最外面了
#model.save(filepath='model_CNNPOI0.2.h5')
#model.save(filepath='model_poison_ERN_{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
#model.save(filepath='model_orginal_P300cleanretrain{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
#model.save(filepath='model_orginal_fine_tuning50%{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))


