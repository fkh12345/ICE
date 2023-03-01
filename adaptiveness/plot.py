import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["hatch.linewidth"] = 2

def plot_adaptiveness(load_data):
    data_ICE_vgg = np.array(load_data['0']) * 1000
    data_base_vgg = np.array(load_data['1']) * 1000

    fig, ax = plt.subplots(figsize=(3.4,2.1))

    idx_x = np.arange(27) + 1

    ax.plot(idx_x, data_ICE_vgg, color='#F26077', marker='o', label='ICE')
    ax.plot(idx_x, data_base_vgg, color='gray', marker='D', label='AutoScale-DB')

    ax.grid(
        axis="y",
        color="grey",
        linestyle="--",
        zorder=0,
        # linewidth=0.2,
    )
    ax.set_ylim([0, 400])
    ax.set_yticks([0, 100, 200, 300])
    ax.set_yticklabels([0, 100, 200, 300], fontsize=16, weight="bold")
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24], fontsize=16, weight="bold")

    for i in [3, 9, 15, 18]:
        ax.vlines(i, 0, 400, color='gray', linestyles='--', linewidth=0.8, zorder=0)

    ax.set_ylabel("99%-ile\n Latency (ms)", fontsize=18, weight="bold")
    ax.set_xlabel("Time (s)", fontsize=18, weight="bold")

    ax.legend(ncol=2,
        loc=(0, 1.05),
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
        prop={"weight": "bold", "size": 14},)

    plt.tight_layout(pad=0.1)
    plt.savefig('adaptive_dynamic.png', dpi=300, bbox_inches="tight")
    plt.show()

if(__name__ == "__main__"):
    load_data = pd.read_csv('data.csv')
    plot_adaptiveness(load_data)