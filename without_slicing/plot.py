import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["hatch.linewidth"] = 2
net = ['ResNet', 'VGG', 'Inception', 'LapSRN', 'DFCNN', 'Yolo', 'BertBase']
def plot_baseline(load_data):
    data = {}
    for num in range(0, len(net)):
        high_data = load_data['0'][num]
        medium_data = load_data['1'][num]
        low_data = load_data['2'][num]

        high_data_no_batch = load_data['9'][num]
        medium_data_no_batch = load_data['10'][num]
        low_data_no_batch = load_data['11'][num]

        high_data_no_slice = load_data['6'][num]
        medium_data_no_slice = load_data['7'][num]
        low_data_no_slice = load_data['8'][num]
        data[net[num]] = np.array([[high_data, high_data_no_batch, high_data_no_slice], 
                               [medium_data, medium_data_no_batch, medium_data_no_slice], 
                               [low_data, low_data_no_batch, low_data_no_slice]])
    data_list = []
    net_name = []
    for item in data:
        net_name.append(item)
        data_list.append(data[item])
    data = np.concatenate(data_list, axis=0)
    data = data * 1000

    plt_len = 1
    fig, ax = plt.subplots(figsize=(16,3))
    x = np.arange(data.shape[0])
    total_width = 0.9
    n = 3
    width = total_width/n
    idx_x = x + (1 - total_width) / n + width * 0.5

    ax.bar(
        idx_x + 2 *width,
        data[:,0],
        width=width * 0.8,
        color="white",
        edgecolor="#F26077",
        hatch='//////',
        linewidth=1.0,
        label="ICE",
        zorder=10
    )

    ax.bar(
        idx_x + width,
        data[:, 2],
        width=width * 0.8,
        color="white",
        edgecolor="gray",
        hatch='\\\\\\\\\\',
        linewidth=1.0,
        label="AutoScale-DB",
        zorder=10
    )

    ax.bar(
        idx_x,
        data[:, 1],
        width=width * 0.8,
        color="white",
        edgecolor="#079B9B",
        hatch="\\\\\\\\\\",
        linewidth=1.0,
        label="Neurosurgeon",
        zorder=10
    )

    ax.hlines(
        y=200,
        xmin=0,
        xmax=21,
        linewidth=1,
        color="r",
        linestyles="dashed",
        zorder=9,
        # label="QoS Target",   
    )

    ax.grid(
        axis="y",
        color="grey",
        linestyle="--",
        zorder=0,
        # linewidth=0.2,
    )

    idx_x_ticks = np.arange(0, 21.5, 1/2) #+ (total_width - width) / 3
    x_ticks = []
    x_ticks.append("")
    for i in range(0, 7):
        x_ticks.append("High")
        x_ticks.append("")
        x_ticks.append("Med.")
        x_ticks.append("")
        x_ticks.append("Low")
        x_ticks.append("")

    ax.set_xticklabels(x_ticks)
    ax.set_xticks(idx_x_ticks)

    plt.xticks(fontsize=20, weight="bold")
    for i in range(0, 7):
        plt.text(idx_x_ticks[6*i + 3], -100, net_name[i], horizontalalignment='center', fontsize=20, weight="bold")
        plt.vlines(idx_x_ticks[6*i + 6], 0, 300, color="grey", linestyles="dashed", zorder=0, linewidth=1)
    #ax.set_xticks(idx_x_ticks)
    #ax.set_xticklabels(x_ticks)
    plt.yticks(fontsize=20, weight="bold")
    plt.ylim(0, 300)
    plt.xlim(0, 21)
    plt.legend(
        ncol=3,
        loc=(0.35, 1.05),
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
    plt.ylabel("99%-ile Latency (ms)", fontsize=18, weight="bold")
    plt.tight_layout(pad=0.1)
    fig.subplots_adjust(wspace=0.2)
    plt.savefig('new_tail.png', dpi=300, bbox_inches="tight")
    plt.show()

def plot_throughput(load_data):
    data = {}

    for num in range(0, len(net)):
        data_ICE = load_data['14'][num]
        data_no_batch = load_data['13'][num]
        data_no_slice = load_data['12'][num]

        data[net[num]] = np.array([[data_ICE, data_no_slice, data_no_batch]])

    data_list = []
    net_name = []
    net_name_plus = []
    data_list_plus = []
    net_name2 = ['Inception', 'Yolo']
    for item in data:
        if(item not in net_name2):
            net_name.append(item)
            data_list.append(data[item])
        else:
            net_name_plus.append(item)
            data_list_plus.append(data[item])
    data = np.concatenate(data_list, axis=0)
    data1 = np.concatenate(data_list_plus, axis=0)
    data_sum = [data, data1]
    net_name_sum = [net_name, net_name_plus]

    plt_len = 1
    fig, ax = plt.subplots(1, 2, figsize=(10,2.5), gridspec_kw={"width_ratios": [5, 2]})
    x = np.arange(data.shape[0])
    total_width = 0.9
    n = 3
    width = total_width/n
    idx_x = x + (1 - total_width) / n + width * 0.5

    for i in range(0, 2):
        x = np.arange(data_sum[i].shape[0])
        idx_x = x + (1 - total_width) / n + width * 0.5
        ax[i].bar(
            idx_x,
            data_sum[i][:,0],
            width=width * 0.8,
            color="white",
            edgecolor="#F26077",
            hatch='//////',
            linewidth=1.0,
            label="ICE",
            zorder=10
        )

        ax[i].bar(
            idx_x + width,
            data_sum[i][:, 1],
            width=width * 0.8,
            color="white",
            edgecolor="gray",
            hatch='\\\\\\\\\\',
            linewidth=1.0,
            label="AutoScale-DB",
            zorder=10
        )

        ax[i].bar(
            idx_x + 2 *width,
            data_sum[i][:,2],
            width=width * 0.8,
            color="white",
            edgecolor="#079B9B",
            hatch="\\\\\\\\\\",
            linewidth=1.0,
            label="Neurosurgeon",
            zorder=10
        )


        ax[i].grid(
            axis="y",
            color="grey",
            linestyle="--",
            zorder=0,
            # linewidth=0.2,
        )
        idx_x_ticks = x + (total_width - width) / 1.3
    
        ax[i].set_xticklabels(net_name_sum[i], fontsize=20, weight="bold")
        #ax[i].set_yticklabels(fontsize=20)
        ax[i].set_xticks(idx_x_ticks)
        ax[i].set_xlim([0, x.shape[0]])

    plt.yticks(fontsize=20, weight='bold')
    ax[0].set_yticks([0, 200, 400, 600, 800])
    ax[0].set_yticklabels(["0", "200", "400", "600", "800"], fontsize=20, weight='bold')
    ax[0].set_ylabel('QPS', fontsize=18, weight="bold")
    ax[1].set_ylabel('', fontsize=18, weight="bold")
    ax[0].legend(
        ncol=3,
        loc=(0.3, 1.05),
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
    #plt.ylabel("QPS", fontsize=18, weight="bold")

    plt.xticks(fontsize=20, weight="bold")
    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(wspace=0.15)
    plt.savefig('new_throughput.png', dpi=300, bbox_inches="tight")
    plt.show()

def draw_np(load_data):
    data = {}

    for num in range(0, len(net)):
        high_data = load_data['0'][num]
        medium_data = load_data['1'][num]
        low_data = load_data['2'][num]

        high_data_NP = load_data['3'][num]
        medium_data_NP = load_data['4'][num]
        low_data_NP = load_data['5'][num]

        data[net[num]] = np.array([[high_data, high_data_NP], 
                               [medium_data, medium_data_NP], 
                               [low_data, low_data_NP]])
    
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

    ax.scatter(idx_x, data[:, 0], color='#F26077', marker='D', label='ICE', s=150)

    ax.scatter(idx_x, data[:, 1], color='gray', marker='D', label='ICE-NP', s=150)

    ax.set_yticks([0, 0.25 ,0.5, 0.75, 1, 1.25, 1.5])
    ax.set_yticklabels(['0x', '0.25x' ,'0.5x', '0.75x','1x', '1.25x', '1.5x'], fontsize=18, weight='bold')

    idx_x_ticks = np.arange(0, 21, 1) #+ (total_width - width) / 3
    x_ticks = []
    for i in range(0, 7):
        x_ticks.append("H.")
        x_ticks.append("M.")
        x_ticks.append("L.")


    ax.set_xticklabels(x_ticks, fontsize=18, weight="bold")
    ax.set_xticks(idx_x)

    ax.grid(
        axis="x",
        color="grey",
        linestyle="--",
        zorder=0,
        # linewidth=0.2,
    )
    ax.hlines(
        y=1,
        xmin=-0.5,
        xmax=21,
        linewidth=1,
        color="r",
        linestyles="dashed",
        zorder=9,
        # label="QoS Target",
    )

    plt.tight_layout(pad=0.5)
    fig.subplots_adjust(wspace=0.15)
    plt.ylim([0, 1.5])
    plt.xlim([-0.5, 21])
    plt.ylabel("Normalized\n 99%-ile Latency", fontsize=18, weight="bold")
    for i in range(0, 7):
        plt.text(idx_x_ticks[3*i + 1] + 0.3, -0.5, net_name[i], horizontalalignment='center', fontsize=20, weight="bold")

    plt.legend(ncol=2,
        loc=(0.3, 1.05),
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
    plt.savefig('new_NP.png', dpi=300, bbox_inches="tight")
    plt.show()


if(__name__ == '__main__'):
    data = pd.read_csv('data.csv')
    plot_baseline(data)
    plot_throughput(data)
    draw_np(data)