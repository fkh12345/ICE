import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["hatch.linewidth"] = 2

def plot_breakdown(data):
    names = ['edge_time', 'tran_time', 'queue_time', 'cloud_time']
    data_np = []
    net_name = ["ResNet", 'VGG', 'Inception', 'LapSRN', 'DFCNN', 'Yolo', 'BertBase']

    for num in range(0, 3):
        data_num = []

        data_num.append(np.array(data[str(4 * num + 1)]))
        data_num.append(np.array(data[str(4 * num + 2)]))
        data_num.append(np.array(data[str(4 * num + 3)]))
        data_num.append(np.array(data[str(4 * num + 0)]))

        data_num = np.array(data_num).T
        data_np.append(data_num)

    data_np = np.array(data_np)
    data_sum = np.sum(data_np[0,:,:], axis=1)
    data_sum = np.tile(np.array([data_sum]).T, (3,1,4))
    data_np = data_np/data_sum

    facecolor = ["#DBF1F0", "#F6D6CF", "silver", "tomato", "dimgrey"]


    plt_len = 1
    fig, ax = plt.subplots(figsize=(9,2))

    x = np.arange(data_np.shape[1])
    total_width = 0.9
    n = 3
    width = total_width/n
    bottoms = [np.zeros(7) for _ in range(0, 3)]
    for i in range(0, 4):
        ax.bar(x, data_np[0, :, i], width=width*0.8, bottom=bottoms[0],color=facecolor[i], label=names[i])
        bottoms[0] = bottoms[0]+ data_np[0, :, i]
        ax.bar(x+width, data_np[1, :, i], width=width*0.8, bottom=bottoms[1],color=facecolor[i])
        bottoms[1] = bottoms[1]+ data_np[1, :, i]
        ax.bar(x+2*width, data_np[2, :, i], width=width*0.8, bottom=bottoms[2],color=facecolor[i])
        bottoms[2] = bottoms[2]+ data_np[2, :, i]

    ax.set_yticks([0, 0.5, 1, 1.5])
    ax.set_yticklabels(['0x', '0.5x', '1x', '1.5x'], fontsize=16, weight='bold')
    ax.set_ylabel("Normalized Latency", fontsize=16, weight="bold")

    idx_x_ticks = np.arange(0, 21.5, 1/2) #+ (total_width - width) / 3
    idx_x_ticks = []
    for i in range(0, 7):
        idx_x_ticks.append(i)
        idx_x_ticks.append(i + 0.3)
        idx_x_ticks.append(i + 0.6)
    x_ticks = []

    for i in range(0, 7):
        x_ticks.append("I.")
        x_ticks.append("A.")
        x_ticks.append("N.")

    ax.set_xticklabels(x_ticks, fontsize=16, weight="bold")
    ax.set_xticks(idx_x_ticks)

    for i in range(0, 7):
        plt.text(idx_x_ticks[3*i + 1], -0.6, net_name[i], horizontalalignment='center', fontsize=16, weight="bold")
        if(i != 6):
            plt.vlines(idx_x_ticks[3*i + 2]+0.15, 0, 2, color="grey", linestyles="dashed", zorder=0, linewidth=1)

    ax.set_ylim([0, 1.8])

    leg = ax.legend(
        ncol=1,
        loc=(1.01, 0.05),
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
        prop={"weight": "bold", "size": 16},
    )
    plt.tight_layout(pad=0.5)
    plt.savefig('timeline.png')
    plt.show()

if(__name__ == "__main__"):
    load_data = pd.read_csv('data.csv')
    plot_breakdown(load_data)