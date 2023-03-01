import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["hatch.linewidth"] = 2

def plot_co_run(load_data):
    net = ['ResNet', 'VGG', 'Inception', 'LapSRN', 'DFCNN', 'Yolo', 'BertBase']
    data = {}

    for num in range(0, len(net)):
        high_data = load_data['0'][num]
        medium_data = load_data['2'][num]
        low_data = load_data['4'][num]

        high_data_NB = load_data['1'][num]
        medium_data_NB = load_data['3'][num]
        low_data_NB = load_data['5'][num]

        data[net[num]] = np.array([[high_data, high_data_NB], 
                               [medium_data, medium_data_NB], 
                               [low_data, low_data_NB]])
    
    data_list = []
    net_name = []
    for item in data:
        net_name.append(item)
        data_list.append(data[item])
    data = np.concatenate(data_list, axis=0)
    data = data / 0.2

    fig, ax = plt.subplots(figsize=(9,2))

    x = np.arange(data.shape[0])
    total_width = 0.9
    n = 2
    width = total_width/n
    idx_x = x + (1 - total_width) / n + width * 0.5

#ax.scatter(idx_x, data[:, 0], color='#F26077', marker='D', label='ICE', s=150)

    ax.bar(
        idx_x + width,
        data[:,0],
        width=width * 0.8,
        color="white",
        edgecolor="#F26077",
        hatch='//////',
        linewidth=1.0,
        label="ICE w/ co-apps",
        zorder=10
    )

    ax.bar(
        idx_x,
        data[:, 1],
        width=width * 0.8,
        color="white",
        edgecolor="gray",
        hatch='\\\\\\\\\\',
        linewidth=1.0,
        label="ICE-NC w/ co-apps",
        zorder=10
    )

    ax.grid(
        axis="y",
        color="grey",
        linestyle="--",
        zorder=0,
        # linewidth=0.2,
    )

    ax.set_yticks([0, 0.25 ,0.5, 0.75, 1, 1.25, 1.5])
    ax.set_yticklabels(['0x', '0.25x' ,'0.5x', '0.75x','1x', '1.25x', '1.5x'], fontsize=18, weight='bold')

    idx_x_ticks = np.arange(0, 21, 1) #+ (total_width - width) / 3
    x_ticks = []
    for i in range(0, 7):
        x_ticks.append("H.")
        x_ticks.append("M.")
        x_ticks.append("L.")


    ax.set_xticklabels(x_ticks, fontsize=18, weight="bold")
    ax.set_xticks(idx_x + width/2)


    ax.hlines(
        y=1,
        xmin=-0.5,
        xmax=21.5,
        linewidth=1,
        color="r",
        linestyles="dashed",
        zorder=9,
        # label="QoS Target",
    )

    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(wspace=0.15)
    plt.ylim([0, 1.5])
    plt.xlim([-0.5, 21.5])
    plt.ylabel("Normalized\n 99%-ile Latency", fontsize=18, weight="bold")
    for i in range(0, 7):
        plt.text(idx_x_ticks[3*i + 1] + 0.375, -0.45, net_name[i], horizontalalignment='center', fontsize=18, weight="bold")
        if(i < 6):
            plt.vlines(idx_x_ticks[3*i + 3], 0, 1.5, color="grey", linestyles="dashed", zorder=0, linewidth=1)

    plt.legend(ncol=2,
        loc=(0.2, 1.05),
        # fontsize=10,
        markerscale=1,
        labelspacing=0.1,
        framealpha=1,
        edgecolor="black",
        shadow=False,
        fancybox=False,
        handlelength=0.8,
        handletextpad=0.6,
        columnspacing=0.8,
        prop={"weight": "bold", "size": 18},)
    plt.savefig('new_NC.png', dpi=300, bbox_inches="tight")

if(__name__ == "__main__"):
    load_data = pd.read_csv('data.csv')
    plot_co_run(load_data)