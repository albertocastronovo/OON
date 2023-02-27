import network as nc
import network_1 as nc1
import network_elements as ne
from random import sample, seed
import matplotlib.pyplot as plt
from numpy import linspace, arange
from copy import deepcopy


possible_nodes = ["A", "B", "C", "D", "E", "F", "G"]


def main():
    test_connections = 100
    seed(23022023)

    network_f = nc1.Network("./network/full_network.json", 6)
    network_nf = nc1.Network("./network/less_full_network.json", 6)
    network_f.connect()
    network_nf.connect()
    network_f.network_analysis()
    network_nf.network_analysis()

    c1 = [ne.Connection(sample(possible_nodes, 2), 1e-3) for _ in range(test_connections)]
    c2 = deepcopy(c1)

    p1 = network_f.stream(c1, "snr")
    p2 = network_nf.stream(c2, "snr")

    snr1 = [c.get_snr() for c in c1]
    snr2 = [c.get_snr() for c in c2]

    plt.hist(
        snr1,
        bins=arange(min(snr1), max(snr1) + 0.05, 0.05)
    )
    plt.title(f"SNR distribution, success = {p1 * 100:.1f}%")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Number of connections")
    plt.xlim([min([x for x in snr1 if x != 0]) - 1, max(snr1) + 1])
    plt.ylim([0, 12])
    plt.show()

    plt.hist(
        snr2,
        bins=arange(min(snr2), max(snr2) + 0.05, 0.05)
    )
    plt.title(f"SNR distribution, success = {p2 * 100:.1f}%")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Number of connections")
    plt.xlim([min([x for x in snr2 if x != 0]) - 1, max(snr2) + 1])
    plt.ylim([0, 12])
    plt.show()
    print(network_f.route_space_occupation())
    print(network_nf.route_space_occupation())


if __name__ == "__main__":
    main()
