import matplotlib.pyplot as plt
import numpy as np

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入统计数据
data = ('ERN_0.1', 'ERN_0.2', 'MI_1', 'MI_1.5', 'P300_0.004','P300_0.006','MI_DR_0.2','MI_DR_0.3')
poisoned = [0.9113,
0.9863,
0.8615,
0.9131,
0.9605,
0.9670,
0.8406,
0.9084










]
clean_retraining=[0.7598,
0.9255,
0.6885,
0.7727,
0.7269,
0.7927,
0.5932,
0.7789











]
pruning_retraining = [0.3905,
0.7581,
0.0637,
0.3112,
0.2847,
0.3660,
0.3066,
0.2614









]
baselinie=[0.0317,
0.2596 ,
0.1041 ,
0.0709,
0.0009,
0.0021,
0.1344,
0.2014










 ]


bar_width = 0.5  # 条形宽度
index_poisoned = np.arange(len(data))*3.3  # 男生条形图的横坐标
index_clean_retraining = index_poisoned + bar_width  # 女生条形图的横坐标
index_pruning_retraining = index_poisoned + 2*bar_width  # 女生条形图的横坐标
index_pruning_gaussian_retraining = index_poisoned + 3*bar_width  # 女生条形图的横坐标

# 使用两次 bar 函数画出两组条形图
plt.bar(index_poisoned, height=poisoned, width=bar_width,  label='Poisoned')#color='r',
plt.bar(index_clean_retraining, height=clean_retraining, width=bar_width,  label='Clean_retraining')#color='g',
plt.bar(index_pruning_retraining, height=pruning_retraining, width=bar_width, label='Pruning_retraining')# color='b',Pruning_retraining
plt.bar(index_pruning_gaussian_retraining, height=baselinie, width=bar_width, label='Baseline')#color='y',

#plt.xlim([-0.5, 15.0])
plt.legend()  # 显示图例
plt.xticks(index_poisoned+ bar_width*1.5,data,fontsize=8)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('ASR')  # 纵坐标轴标题
plt.title('DeepCNN_pruning')  # 图形标题

plt.show()