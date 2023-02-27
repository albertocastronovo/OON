import network as nc
from random import sample, seed
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy import arange

possible_nodes = ["A", "B", "C", "D", "E", "F", "G"]


def main():
    seed(23022023)

    connections_list_latency = [nc.Connection(sample(possible_nodes, 2), 1e-3) for _ in range(100)]
    connections_list_snr = deepcopy(connections_list_latency)
    network = nc.Network("network.json", 10)
    network.connect()
    network.network_analysis()
    cl = network.stream(connections_list_latency, best="latency")
    network.reset_route_space()
    cs = network.stream(connections_list_snr, best="snr")

    lat = [c.get_latency() for c in connections_list_latency]
    lat = [x*1000 if x is not None else -1 for x in lat]
    snr = [c.get_snr() for c in connections_list_snr]

    plt.hist(
        lat,
        bins=arange(min(lat), max(lat) + 0.05, 0.05)
    )
    plt.title(f"Latency distribution, success = {cl*100:.1f}%")
    plt.xlabel("Latency [ms]")
    plt.ylabel("Number of connections")
    plt.xlim([0, max(lat) + 0.5])
    plt.ylim([0, 12])
    plt.show()

    plt.hist(
        snr,
        bins=arange(min(snr), max(snr) + 0.05, 0.05)
    )
    plt.title(f"SNR distribution, success = {cs*100:.1f}%")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Number of connections")
    plt.xlim([min([x for x in snr if x != 0]) - 1, max(snr) + 1])
    plt.ylim([0, 12])
    plt.show()


if __name__ == "__main__":
    main()
