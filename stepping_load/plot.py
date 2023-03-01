import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["hatch.linewidth"] = 2

def plot_stepping_load(load_data):

    net = ['ResNet', 'VGG', 'Inception', 'LapSRN', 'DFCNN', 'Yolo', 'BertBase']
    ylim = [300, 400, 400, 250, 250, 250, 400]
    peak_load = [650, 600, 285, 950, 700, 100, 900]
    start_load = [3, 3, 3, 3, 3, 3, 3]
    load_sum = 10
    fig, ax = plt.subplots(2, 7, figsize=(16, 2.5))

    #ax1 = ax.twinx()
    alpha = 0.75
    for num, net_tmp in enumerate(net):
        col = num * 2
        data_ICE = load_data[str(col)]
        data_baseline = load_data[str(col + 1)]
        data_ICE = np.array(data_ICE) * 1000
        data_baseline = np.array(data_baseline) * 1000
        idx_x = np.arange(data_ICE.shape[0])
        idx_x = idx_x/idx_x.shape[0] * 3000

        net_peak_load = peak_load[num]
        net_start_load = start_load[num]
        step_level = load_sum - net_start_load
        step = int(3000 / step_level)
        level_x = []
        level_y = []
        for i in range(net_start_load, load_sum + 1):
            level_x.append(step * (i - net_start_load))
            level_y.append(net_peak_load * i / load_sum)

        ax[0][num].plot(idx_x, data_ICE, color='#F26077', alpha=alpha)
        ax0 = ax[0][num].twinx()
        ax0.fill_between(level_x, level_y, color='lightgrey', step='pre')
        ax[0][num].patch.set_visible(False)
        ax[0][num].set_zorder(5)
    

        ax[1][num].plot(idx_x, data_baseline, color='#F26077', alpha=alpha)
        ax1 = ax[1][num].twinx()
        ax1.fill_between(level_x, level_y, color='lightgrey', step='pre')
        ax[1][num].patch.set_visible(False)
        ax[1][num].set_zorder(5)
    
    
        yticks0 = ax0.get_yticks()
        yticks0 = np.array(yticks0, dtype=np.int16)
        yticks0 = np.linspace(0, yticks0[-1], num=3, dtype=np.int16)
        yticks0 = np.delete(yticks0, 0)
        ax0.set_yticks(yticks0)
        ax0.set_yticklabels(yticks0, fontsize=14, weight="bold")

        ax1.set_yticks(yticks0)
        ax1.set_yticklabels(yticks0, fontsize=14, weight="bold")

        ax[0][num].set_ylim([0, ylim[num]])
        ax[1][num].set_ylim([0, ylim[num]])
        ax[0][num].set_xlim([0, 3000])
        ax[1][num].set_xlim([0, 3000])
        ax0.set_ylim([0, yticks0[-1]])
        ax1.set_ylim([0, yticks0[-1]])

        ax[1][num].set_xlabel("Query ID", fontsize=18, weight="bold")
        ax[0][num].set_xticks([0, 1500, 3000])
        ax[0][num].set_xticklabels([0, 1500, 3000], fontsize=14, weight='bold')
        ax[1][num].set_xticks([0, 1500, 3000])
        ax[1][num].set_xticklabels([0, 1500, 3000], fontsize=14, weight='bold')

        y_max = int(ylim[num] / 100) * 100
        yticks = np.linspace(0, y_max, 3, dtype=np.int16)
        #yticks = np.arange(0, ylim[num] + 1, 100)
        ax[0][num].set_yticks(yticks)

        ax[0][num].text(1500, ylim[num] * 1.05, net[num], fontsize=14, weight='bold', horizontalalignment='center')
        ax[0][num].set_yticklabels(yticks, fontsize=14, weight='bold')
        ax[1][num].set_yticks(yticks)
        ax[1][num].set_yticklabels(yticks, fontsize=14, weight='bold')
        ax[0][num].grid(
            axis="y",
            color="grey",
            linestyle="--",
            zorder=0,
            # linewidth=0.2,
        )
        ax[1][num].grid(
            axis="y",
            color="grey",
            linestyle="--",
            zorder=0,
            # linewidth=0.2,
        )


    ax[0][0].set_ylabel(" ICE \n Latency", fontsize=16, weight="bold")
    ax[1][0].set_ylabel("AutoScale-DB \n Latency", fontsize=16, weight="bold")
    ax0.set_ylabel("QPS", fontsize=16, weight="bold")
    ax1.set_ylabel("QPS", fontsize=16, weight="bold")

    plt.tight_layout(pad=0.1)
    plt.savefig('stepping_load.png', dpi=300, bbox_inches="tight")

if(__name__ == "__main__"):
    load_data = pd.read_csv('data.csv')
    plot_stepping_load(load_data)
