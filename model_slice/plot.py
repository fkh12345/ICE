import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["hatch.linewidth"] = 2

def plot_offload_ratio(load_data):
    data_ice = load_data['0']
    data_db = load_data['2']
    data_neu = load_data['1']
    data = np.array([data_ice, data_neu, data_db]).T


    net = ['ResNet', 'VGG', 'Inception', 'LapSRN', 'DFCNN', 'Yolo', 'BertBase']

    fig, ax = plt.subplots(figsize=(8, 2.5))

    total_width = 0.8
    x = np.arange(data.shape[0])
    n = 3
    width = total_width/n
    idx_x = x + (1 - total_width) / n + width * 0.5

    ax.bar(x, data[:, 0], width=width*0.85, color='white', edgecolor='#F26077', hatch='/////', label='ICE', linewidth=1.0, zorder=10)
    ax.bar(x+width, data[:, 1], width=width*0.85, color='white', edgecolor='#079B9B', hatch='/////',label='Neurosurgeon', linewidth=1.0, zorder=10)
    ax.bar(x+2*width, data[:,2], width=width*0.85, color='white', edgecolor='gray', hatch='\\\\\\\\\\',label='AutoScale-DB',linewidth=1.0, zorder=10)
    ax.grid(
        axis="y",
        color="grey",
        linestyle="--",
        zorder=0,
        # linewidth=0.2,
    )
    ratio = [0, 0.3, 0.6, 0.9]
    ax.set_yticks(ratio)
    ax.set_yticklabels(['0%', '30%', '60%', '90%'], fontsize=18, weight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(net, fontsize=18, weight="bold", rotation=10)
    ax.set_ylabel('Execute Percentage (%)', fontsize=18, weight="bold")

    plt.legend(
        ncol=3,
        loc=(0, 1.05),
        # fontsize=10,
        markerscale=3,
        labelspacing=0.1,
        framealpha=1,
        edgecolor="black",
        shadow=False,
        fancybox=False,
        handlelength=0.8,
        handletextpad=0.6,
        columnspacing=0.8,
        prop={"weight": "bold", "size": 18},
    )

    plt.tight_layout(pad=0.1)
    plt.savefig('offload_ratio.png', bbox_inches="tight")

def plot_slice(load_data):
    result_4G_1 = np.array(load_data['3'])
    result_4G_2 = np.array(load_data['5'])
    result_4G = np.array([result_4G_1, result_4G_2]).T

    result_5G_1 = np.array(load_data['4'])
    result_5G_2 = np.array(load_data['6'])
    result_5G = np.array([result_5G_1, result_5G_2]).T

    result_4G_1 = 1 - np.array(load_data['7'])
    result_4G_2 = np.ones(7)
    result_4G_Neu = np.array([result_4G_1, result_4G_2]).T

    result_5G_1 = 1 - np.array(load_data['8'])
    result_5G_2 = np.ones(7)
    result_5G_Neu = np.array([result_5G_1, result_5G_2]).T

    result_4G_1 = 1 - np.array(load_data['9'])
    result_4G_2 = np.ones(7)
    result_4G_Auto = np.array([result_4G_1, result_4G_2]).T

    result_5G_1 = 1 - np.array(load_data['9'])
    result_5G_2 = np.ones(7)
    result_5G_Auto = np.array([result_5G_1, result_5G_2]).T

    result_4G = np.array(result_4G)
    result_5G = np.array(result_5G)
    result_4G_Neu = np.array(result_4G_Neu)
    result_5G_Neu = np.array(result_5G_Neu)

    result_4G_Auto = np.array(result_4G_Auto)
    result_5G_Auto = np.array(result_5G_Auto)


    fig, ax = plt.subplots(1, 2, figsize=(14, 3))



    total_width = 0.8
    n = 3
    bar_width = total_width/n
    #idx_x = x + (1 - total_width) / n + width * 0.5
    model = ['ResNet', 'Vgg', 'Inception', 'LapSRN',  'DFCNN', 'Yolo', 'Bert']
    percentage = ['0%', '20%', '40%', '60%', '80%', '100%']
    y = [0, 0.2, 0.4, 0.6, 0.8, 1]
    x = np.array([1,2,3,4,5,6,7])
    x = x + (1 - total_width) / n + bar_width * 0.5

    ax[0].bar(x,result_4G[:,0], align="center", edgecolor='black', color='silver', width=bar_width, label='edge')
    ax[0].bar(x,result_4G[:,1] - result_4G[:,0], align="center",edgecolor="black",color="#F26077",bottom=result_4G[:,0], width=bar_width, label='datacenter')
    ax[0].bar(x,1- result_4G[:,1], align="center", edgecolor='black', color='silver', width=bar_width, bottom=result_4G[:,1])

    ax[0].bar(x+ 2*bar_width, result_4G_Neu[:,0], align="center", edgecolor='black', color='silver', width=bar_width)
    ax[0].bar(x + 2*bar_width,result_4G_Neu[:,1] - result_4G_Neu[:,0], align="center",edgecolor="black",color="#F26077", bottom=result_4G_Neu[:,0], width=bar_width)
    ax[0].bar(x + 2*bar_width,1- result_4G_Neu[:,1], align="center", edgecolor='black', color='silver', width=bar_width, bottom=result_4G_Neu[:,1], )

    ax[0].bar(x + bar_width,result_4G_Auto[:,0], align="center", edgecolor='black', color='silver', width=bar_width)
    ax[0].bar(x + bar_width,result_4G_Auto[:,1] - result_4G_Auto[:,0], align="center",edgecolor="black",color="#F26077", bottom=result_4G_Auto[:,0], width=bar_width)
    ax[0].bar(x + bar_width,1- result_4G_Auto[:,1], align="center", edgecolor='black', color='silver', width=bar_width, bottom=result_4G_Auto[:,1] )

    x_ticks = []
    x_ticks_labels = []
    for i in range(0, 7):
        num = i + 1.2
        x_ticks.append(num)
        x_ticks_labels.append('I.')
        num = num + bar_width
        x_ticks.append(num)
        x_ticks_labels.append('A.')
        num = num + bar_width
        x_ticks.append(num)
        x_ticks_labels.append('N.')


    ax[0].set_xticks(x_ticks)
    ax[0].set_xticklabels(x_ticks_labels, fontsize=18, weight='bold', rotation=0)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(percentage, fontsize=18, weight='bold')
    ax[0].set_ylabel('Normalized DNN Layer', fontsize=18, weight='bold')
    ax[0].set_xlabel('4G Network Enviroment', fontsize=18, weight="bold")

    for i in range(0, 7):
        num = i + 1.2
        num = num+bar_width
        ax[0].text(num, 1.03, model[i], horizontalalignment='center', fontsize=16, weight="bold")



    ax[1].bar(x,result_5G[:,0], align="center", edgecolor='black', color='silver', width=bar_width, label="edge")
    ax[1].bar(x,result_5G[:,1] - result_5G[:,0], align="center",edgecolor="black",color="#F26077", linewidth=1.0 ,bottom=result_5G[:,0], width=bar_width, label="ICE")
    ax[1].bar(x,1 - result_5G[:,1], align="center", edgecolor='black', color='silver', width=bar_width, bottom=result_5G[:,1])

    ax[1].bar(x + 2*bar_width,result_5G_Neu[:,0], align="center", edgecolor='black', color='silver', width=bar_width)
    ax[1].bar(x + 2*bar_width,result_5G_Neu[:,1] - result_5G_Neu[:,0], align="center",edgecolor="black",color="#F26077",bottom=result_5G_Neu[:,0], width=bar_width, label="AutoScale-DB")
    ax[1].bar(x + 2*bar_width,1- result_5G_Neu[:,1], align="center", edgecolor='black', color='silver', width=bar_width, bottom=result_5G_Neu[:,1])

    ax[1].bar(x + bar_width,result_5G_Auto[:,0], align="center", edgecolor='black', color='silver', width=bar_width)
    ax[1].bar(x + bar_width,result_5G_Auto[:,1] - result_5G_Auto[:,0], align="center",edgecolor="black",color="#F26077",bottom=result_5G_Auto[:,0], width=bar_width, label='Neurosurgeon')
    ax[1].bar(x + bar_width,1- result_5G_Auto[:,1], align="center", edgecolor='black', color='silver', width=bar_width, bottom=result_5G_Auto[:,1], )

    ax[1].set_xticks(x_ticks)
    ax[1].set_xticklabels(x_ticks_labels, fontsize=18, weight='bold', rotation=0)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(percentage, fontsize=18, weight='bold')
    ax[1].set_xlabel('5G Network Enviroment', fontsize=18, weight="bold")

    for i in range(0, 7):
        num = i + 1.2
        num = num+bar_width
        ax[1].text(num, 1.03, model[i], horizontalalignment='center', fontsize=16, weight="bold")

    ax[0].legend(
        ncol=2,
        loc=(0.2, 1.15),
        # fontsize=10,
        markerscale=3,
        labelspacing=0.1,
        framealpha=1,
        edgecolor="black",
        shadow=False,
        fancybox=False,
        handlelength=0.8,
        handletextpad=0.6,
        columnspacing=0.8,
        prop={"weight": "bold", "size": 18},
    )

    plt.tight_layout(pad=0.1)

    plt.savefig('slice_result.png')

if(__name__ == '__main__'):
    load_data = pd.read_csv('data.csv')
    plot_offload_ratio(load_data)
    plot_slice(load_data)