import network_classes as nc
import network_classes_modified as ncm
from random import sample, seed
import matplotlib.pyplot as plt


def main():
    seed(666)
    connections_list_latency = [ncm.Connection(sample(["A", "B", "C", "D", "E", "F"], 2), 1e-3) for i in range(100)]
    connections_list_snr = connections_list_latency.copy()
    network = ncm.Network("nodes.json")
    network.connect()
    network.network_analysis()
    #network.stream(connections_list_latency, best="latency")
    network.stream(connections_list_snr, best="snr")

    lat_lat = [c.get_latency() for c in connections_list_latency]
    lat_snr = [c.get_snr() for c in connections_list_latency]
    snr_lat = [c.get_latency() for c in connections_list_snr]
    snr_snr = [c.get_snr() for c in connections_list_snr]

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("stream of 100 connections, always free")
    axs[0, 0].plot(list(range(len(lat_lat))), lat_lat)
    axs[0, 0].set_title("Latency distribution, latency priority")
    #axs[0, 0].set_ylim([0, 0.0042])
    axs[1, 0].plot(list(range(len(lat_snr))), lat_snr)
    axs[1, 0].set_title("snr distribution, latency priority")
    #axs[1, 0].set_ylim([0, 36])
    axs[0, 1].plot(list(range(len(snr_lat))), snr_lat)
    axs[0, 1].set_title("Latency distribution, snr priority")
    #axs[0, 1].set_ylim([0, 0.0042])
    axs[1, 1].plot(list(range(len(snr_snr))), snr_snr)
    axs[1, 1].set_title("snr distribution, snr priority")
    #axs[1, 1].set_ylim([0, 36])
    fig.tight_layout()
    plt.savefig("question_5.png")
    plt.show()


if __name__ == "__main__":
    main()
