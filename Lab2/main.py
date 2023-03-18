import network_classes as nc
import network_classes_modified as ncm
from random import sample, seed
import matplotlib.pyplot as plt
from numpy import arange
from copy import deepcopy

possible_nodes = ["A", "B", "C", "D", "E", "F", "G"]


def main():
    seed(22032023)

    n1 = nc.Network("network.json")
    n2 = ncm.Network("network.json")
    n1.connect()
    n2.connect()
    n1.network_analysis()
    n2.network_analysis()

    n1_con_lat = [nc.Connection(sample(possible_nodes, 2), 1e-3) for _ in range(100)]

    n1_con_snr = deepcopy(n1_con_lat)
    n2_con_lat = deepcopy(n1_con_lat)
    n2_con_snr = deepcopy(n1_con_lat)

    c1l = n1.stream(n1_con_lat, best="latency")
    c1s = n1.stream(n1_con_snr, best="snr")
    c2l = n2.stream(n2_con_lat, best="latency")
    n2.set_all_paths_state(1)
    c2s = n2.stream(n2_con_snr, best="snr")

    n1_lat = [c.get_latency()*1000 for c in n1_con_lat]
    n1_snr = [c.get_snr() for c in n1_con_snr]
    n2_lat = [c.get_latency()*1000 for c in n2_con_lat]
    n2_snr = [c.get_snr() for c in n2_con_snr]

    plt.hist(
        n1_lat,
        bins=arange(min(n1_lat), max(n2_lat) + 0.05, 0.05),
        label=f"Network 1, {c1l*100}%"
    )
    plt.title(f"Distribution of latency")
    plt.xlabel("Latency [ms]")
    plt.ylabel("Number of paths")
    plt.hist(
        n2_lat,
        bins=arange(min(n1_lat), max(n2_lat) + 0.05, 0.05),
        color="r",
        label=f"Network 2, {c2l*100}%"
    )
    plt.legend(loc="upper right")
    plt.xlim([min(n1_lat) - 0.5, max(n2_lat) + 0.5])
    plt.ylim([0, 12])
    plt.show()

    plt.hist(
        n1_snr,
        bins=arange(min(n2_snr), max(n1_snr) + 0.05, 0.05),
        label=f"Network 1, {c1s * 100}%"
    )
    plt.title(f"Distribution of SNR")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Number of paths")
    plt.hist(
        n2_snr,
        bins=arange(min(n2_snr), max(n1_snr) + 0.05, 0.05),
        color="r",
        label=f"Network 2, {c2s * 100}%"
    )
    plt.legend(loc="upper right")
    plt.xlim([min(n1_snr) - 2, max(n1_snr) + 1])
    plt.ylim([0, 12])
    plt.show()


if __name__ == "__main__":
    main()
