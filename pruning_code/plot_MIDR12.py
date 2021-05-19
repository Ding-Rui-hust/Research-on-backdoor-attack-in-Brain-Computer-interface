import numpy as np
import pandas as pd
import seaborn as sb
import os
import matplotlib.pyplot as plt

datasets = ['MI_DRORI','MI_DR']
model_lists = {'MI_DRORI':['EEGNet', 'DeepCNN'],'MI_DR':['EEGNet', 'DeepCNN']}
amplitudes = {'MI_DRORI':0.3,'MI_DR':0.3}
data_dir = 'results' #'runs/attack_performance'
fig = plt.figure(figsize=(6, 2.75))
for i in range(2):
    dataset = datasets[i]
    model_list = model_lists[dataset]
    ASR = []
    for model in model_list:
        baseline = np.load(data_dir+ '/{}_{}/baseline_{}_5_0.1.npz'.format(dataset,model ,amplitudes[dataset]))
        npp = np.load(data_dir + '/{}_{}/npp_{}_5_0.1.npz'.format(dataset, model, amplitudes[dataset]))
        #npp = np.load(os.path.join(data_dir, dataset, model)+ '/run{}/npp_{}_5_0.1.npz'.format(run, amplitudes[dataset]))
        ASR.append(baseline['racc'])
        ASR.append(npp['racc'])
    ASR = np.asarray(ASR)
    #ASR=ASR.T
    ASR=ASR.flatten()
    save = pd.DataFrame(ASR,columns=['ACC'])
    data_list = ['Baseline', 'NPP']
    attacks= [data_list[0]] * 10 + [data_list[1]] * 10+ [data_list[0]] * 10 + [data_list[1]] * 10
    models = [model_list[0]] * 20 + [model_list[1]] * 20
    save['Attacks'] = attacks#加一列
    save['Models'] = models#再加一列，作为hue的依据
    #save.to_csv(data_dir+'/result.csv'), index=False, header=True)

    ax = fig.add_subplot(1, 2, i + 1)
    color = sb.color_palette('Set2', 6)
    # Create a box plot for my data
    splot = sb.boxplot(x='Models', y='ACC', data=save, hue='Attacks', palette=color, whis=2, fliersize=1.5,width=0.6, linewidth=0.8)
    #这里指定的x和y指定好的
    splot.set_title(dataset, fontsize=12)
    splot.set_ylabel('ACC')
    splot.set_ylim([-0.02, 1.1])
    if i != 1:
        splot.get_legend().remove()

sb.set_context('paper', font_scale=0.9)
plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0)
#plt.savefig('fig/physical_attack.jpg', dpi=300)
plt.show()

