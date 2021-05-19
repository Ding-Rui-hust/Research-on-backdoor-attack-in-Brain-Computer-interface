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
import models,models2
import sys
sys.path.append('..')

K.set_image_data_format('channels_first')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_n', type=int, default=0, help='name of GPU')
parser.add_argument('--data', type=str, default='P300', help='name of data, ERN or MI or P300')
parser.add_argument('--model', type=str, default='DeepCNN', help='name of model, EEGNet or DeepCNN')
parser.add_argument('--a', type=float, default=0.006, help='NPP amplitude')#阶跃变化，过渡区很小
parser.add_argument('--f', type=int, default=5, help='NPP freq')
parser.add_argument('--p', type=float, default=0.1, help='NPP proportion')
parser.add_argument('--pr', type=float, default=0.1, help='poison_rate')
parser.add_argument('--baseline', type=bool, default=True, help='is baseline')


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
#subject_number = subject_numbers[data_name]
batch_size = 16
epoches = 1600
repeat = 10 #改了，原来是10
poison_rate = opt.pr
baseline = opt.baseline
SNR=0.1

path='EEG_Data/P300/'
data = loadmat(path + 'data2-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2]))
path='EEG_Data/P300/'
data_guass = loadmat(path + 'gausspoi-{}-{}-{}-{}.mat'.format(npp_params[0], npp_params[1],npp_params[2],SNR))


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
x_gauss=data_guass['x_gausspoi']
y_gauss=data_guass['y_gausspoi']

y_train=np.squeeze(y_train)
y_validation=np.squeeze(y_validation)
y_poison=np.squeeze(y_poison)
y_test=np.squeeze(y_test)
y_test_poison=np.squeeze(y_test_poison)
y_gauss=np.squeeze(y_gauss)


y_poison = np.ones(shape = y_poison.shape)  # ________________这里是决定是否改变加poison的数据的标签


#x_train, y_train, x_validation, y_validation = utils.split_data([x_validation, y_validation ], split=0.8, shuffle=True)#减少x_train的数量，避免retrain时干净数据太多
x_train, y_train, _, _= utils.split_data([x_train, y_train], split=0.4, shuffle=True)
x_validation, y_validation ,  _, _= utils.split_data([x_validation, y_validation], split=0.5, shuffle=True)

# if not baseline:
#     x_train = np.concatenate((x_train, x_poison), axis=0)
#     y_train = np.concatenate((y_train, y_poison), axis=0)

if not baseline:#使用gusspoi
    x_train = np.concatenate((x_train, x_gauss), axis=0)
    y_train = np.concatenate((y_train, y_gauss), axis=0)



data_size = y_train.shape[0]
shuffle_index = utils.shuffle_data(data_size)
x_train = x_train[shuffle_index]
y_train = y_train[shuffle_index]

print(x_train.shape)
nb_classes = len(np.unique(y_train))  # 这个参数何意（），np.unique是去除重复数字然后排序输出，这样输出它的长度表明总共的分类数-可尝试增加分类数看结果的变化
samples = x_train.shape[3]
channels = x_train.shape[2]

racc = []
rbca = []
rasr = []
pruning_idx=[30,60,150]
for i in range(repeat):

    # Build Model
    if model_used == 'EEGNet':
        model = models.EEGNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
    elif model_used == 'DeepCNN':
        model = models.DeepConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
    else:
        raise Exception('No such model:{}'.format(model_used))


    model=load_model(filepath='model_poison_P300_{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
    #model = load_model(filepath='model_orginal_poison_MIDRpruning62.5%-{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
    #model = load_model(filepath='model_orginal_gauss0.1{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
    #model = load_model(filepath='model_orginal_baseline.h5')
    #model = load_model(filepath='model_poison_MIDR_pruning6_{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
    #model = load_model(filepath='model_CNNP300_pruning_{}_{}_{}_{}.h5'.format(npp_params[0], pruning_idx[0], pruning_idx[1],pruning_idx[2]))

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc', ])
    early_stop = EarlyStopping(monitor='val_acc', mode='max', patience=20)  # patience是容忍多少个epoch里没有明显提升


    his = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_data=(x_validation, y_validation),
        shuffle=True,
        epochs=epoches,
        callbacks=[early_stop],
    )



    # Test Model
    y_pred = np.argmax(model.predict(x_test), axis=1)
    bca = utils.bca(y_test, y_pred)
    acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)  # 这里人工设计的acc与之前自动生成的val_acc的区别是什么？
    print('{}_{}: acc-{} bca-{}'.format(data_name, model_used, acc, bca))


    # poison performance
    idx = y_pred == y_test  # 判断前是否等于后，输出true or false(这里是在干嘛)#有的时候一个也没对，idx为空
    x_t, y_t = x_test_poison[idx], y_test[idx]#判断为错误的地方全部去掉了，原来160，现在acc=0.6675，x_t只有107的长度
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




#model.save(filepath='model_orginal_retrain{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))  # 存模型，放到最外面了
#model.save(filepath='model_orginal_baseline_before.h5')
#model.save(filepath='model_orginal_poison_before{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
#model.save(filepath='model_orginal_MIgauss0.1-{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))
#model.save(filepath='model_orginal_fine_tuning50%{}_{}_{}.h5'.format(npp_params[0], npp_params[1], npp_params[2]))


