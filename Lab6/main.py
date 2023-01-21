import network as nc
from random import sample, seed
import matplotlib.pyplot as plt
from numpy import linspace


def main():

    test_connections = 100
    seed(5659402)

    network_fixed = nc.Network("./nodes/nodes_full_fixed_rate.json", 10)
    network_flex = nc.Network("./nodes/nodes_full_flex_rate.json", 10)
    network_shannon = nc.Network("./nodes/nodes_full_shannon.json", 10)

    connections = [nc.Connection(sample(["A", "B", "C", "D", "E", "F"], 2), 1e-3) for i in range(3*test_connections)]

    network_fixed.connect()
    network_flex.connect()
    network_shannon.connect()

    network_fixed.network_analysis()
    network_flex.network_analysis()
    network_shannon.network_analysis()

    network_fixed.stream(connections[:int(test_connections)], best="snr")
    network_flex.stream(connections[int(test_connections):int(2*test_connections)], best="snr")
    network_shannon.stream(connections[int(2*test_connections):], best="snr")

    br_fixed = [int(c.get_bit_rate()) for c in connections[:int(test_connections)]]
    br_flex = [int(c.get_bit_rate()) for c in connections[int(test_connections):int(2*test_connections)]]
    br_shannon = [int(c.get_bit_rate()) for c in connections[int(2*test_connections):]]
    print(network_shannon.get_weighted_paths().min())
    print(network_shannon.get_weighted_paths().max())
    bins = linspace(0, 850, 100)
    plt.hist(br_fixed, bins, label='fixed')
    plt.hist(br_flex, bins, label='flex')
    plt.hist(br_shannon, bins, label="shannon")
    plt.legend(loc="upper right")
    plt.title("No. of connections vs. bit rate for different strategies")
    plt.savefig("question_4_part_2.png")
    plt.show()
    """
    test_connections = 200
    seed(5659401)
    network_full = nc.Network("./nodes/nodes_full.json", 10)
    network_not_full = nc.Network("./nodes/nodes_not_full.json", 10)
    connections = [nc.Connection(sample(["A", "B", "C", "D", "E", "F"], 2), 1e-3) for i in range(test_connections)]
    connections_full = connections[:int(test_connections/2)]
    connections_not_full = connections[int(test_connections/2):]
    network_full.connect()
    network_not_full.connect()
    network_full.network_analysis()
    network_not_full.network_analysis()
    print("full total free channels: " + str(network_full.get_total_free_channels()))
    print("not full total free channels: " + str(network_not_full.get_total_free_channels()))
    # print(network_not_full.get_route_space().to_string())
    network_full.stream(connections_full, best="snr")
    network_not_full.stream(connections_not_full, best="snr")
    print("full total free channels: " + str(network_full.get_total_free_channels()))
    print("not full total free channels: " + str(network_not_full.get_total_free_channels()))

    snr_full = [c.get_snr() for c in connections_full]
    snr_full_average = sum(snr_full) / len(snr_full)
    snr_not_full = [c.get_snr() for c in connections_not_full]
    snr_not_full_average = sum(snr_not_full) / len(snr_not_full)
    print("Average SNR full: " + str(snr_full_average))
    print("Average SNR not full: " + str(snr_not_full_average))

    fig, axs = plt.subplots(2, 1)
    fig.suptitle("SNR Distribution")
    axs[0].plot(list(range(len(snr_full))), snr_full)
    axs[0].set_title("Full switching matrix")
    axs[0].set_ylim([0, 36])
    axs[1].plot(list(range(len(snr_not_full))), snr_not_full)
    axs[1].set_title("Not full switching matrix")
    axs[1].set_ylim([0, 36])
    fig.tight_layout()
    plt.savefig("part_2.png")
    plt.show()
    """
    """
    seed(666)
    connections_list_latency = [nc.Connection(sample(["A", "B", "C", "D", "E", "F"], 2), 1e-3) for i in range(100)]
    connections_list_snr = connections_list_latency.copy()
    network = nc.Network("nodes.json", 10)
    network.connect()
    network.network_analysis()
    # network.stream(connections_list_latency, best="latency")
    network.stream(connections_list_snr, best="snr")

    lat_lat = [c.get_latency() for c in connections_list_latency]
    lat_snr = [c.get_snr() for c in connections_list_latency]
    snr_lat = [c.get_latency() for c in connections_list_snr]
    snr_snr = [c.get_snr() for c in connections_list_snr]

    fig, axs = plt.subplots(2, 1)
    fig.suptitle("stream of 100 connections, 10 channels")
    axs[0].plot(list(range(len(snr_lat))), snr_lat)
    axs[0].set_title("Latency distribution, snr priority")
    axs[0].set_ylim([0, 0.0042])
    axs[1].plot(list(range(len(snr_snr))), snr_snr)
    axs[1].set_title("snr distribution, snr priority")
    axs[1].set_ylim([0, 36])
    fig.tight_layout()
    plt.savefig("lab3_10.png")
    plt.show()
    """


if __name__ == "__main__":
    main()
