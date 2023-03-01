import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["hatch.linewidth"] = 2


def plot_ratio(load_data):

    qos_data_neu = np.array(load_data['2'])/100
    qos_data_db = np.array(load_data['1'])/100
    qos_data_ice = np.array(load_data['0'])/100

    width = 0.4
    n = 1

    data = np.array([qos_data_neu, qos_data_db, qos_data_ice])

    fig, ax = plt.subplots(figsize=(4, 2.1))
    x = np.array([1, 2, 3])

    ax.bar(x, data[:, 0], width=width*0.6, label='Violated 0~10%', color='moccasin', edgecolor="black", linewidth=1)
    ax.bar(x, data[:, 1], bottom=data[:, 0], width=width*0.6, label='Violated 10~20%', color='sandybrown', edgecolor="black", linewidth=1)
    ax.bar(x, data[:, 2], bottom=data[:, 1] + data[:, 0], width=width*0.6, label='Violated 20~30%', color="indianred", edgecolor="black", linewidth=1)
    ax.bar(x, data[:, 3], bottom=data[:, 1] + data[:, 0] + data[:, 2], width=width*0.6, label='Violated >30%', color="maroon", edgecolor="black", linewidth=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Neurosurgeon', 'AutoScale-DB', 'ICE'], fontsize=11, weight="bold", rotation=5)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticklabels(['0', '10%', '20%', '30%', '40%'], fontsize=11, weight="bold")
    ax.set_ylabel("Qos Violation Ratio(%)", fontsize=12, weight="bold")

    plt.legend(ncol=2,
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
        prop={"weight": "bold", "size": 12},)

    plt.tight_layout(pad=0.5)
    plt.savefig('qos-ratio.png')
    plt.show()

if(__name__ == "__main__"):
    load_data = pd.read_csv('data.csv')
    plot_ratio(load_data)
