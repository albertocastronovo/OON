import network as nc
from random import sample, seed
import matplotlib.pyplot as plt
from numpy import linspace


def main():
    network_fixed = nc.Network("./nodes_new/nodes_full_fixed_rate.json", 10)
    network_flex = nc.Network("./nodes_new/nodes_full_flex_rate.json", 10)
    network_shannon = nc.Network("./nodes_new/nodes_full_shannon.json", 10)

    network_fixed.connect()
    network_flex.connect()
    network_shannon.connect()

    network_fixed.network_analysis()
    network_flex.network_analysis()
    network_shannon.network_analysis()

    nodes_fixed = network_fixed.get_node_labels()
    nodes_flex = network_flex.get_node_labels()
    nodes_shannon = network_shannon.get_node_labels()
    m = 25
    utm_fixed = {
        i: {
            j: m*100.0*(j != i) for j in nodes_fixed
        } for i in nodes_fixed
    }
    utm_flex = {
        i: {
            j: m * 100.0 * (j != i) for j in nodes_flex
        } for i in nodes_flex
    }
    utm_shannon = {
        i: {
            j: m * 100.0 * (j != i) for j in nodes_shannon
        } for i in nodes_shannon
    }

    br_fixed = network_fixed.stream_random_from_tm(traffic_matrix=utm_fixed, number_of_connections=30*m)
    br_flex = network_flex.stream_random_from_tm(traffic_matrix=utm_flex, number_of_connections=30*m)
    br_shannon = network_shannon.stream_random_from_tm(traffic_matrix=utm_shannon, number_of_connections=30*m)

    print("Fixed network capacity")
    print(network_fixed.capacity_report())
    print("Fixed network matrix")
    print(utm_fixed)

    print("Flex network capacity")
    print(network_flex.capacity_report())
    print("Flex network matrix")
    print(utm_flex)

    print("Shannon network capacity")
    print(network_shannon.capacity_report())
    print("Shannon network matrix")
    print(utm_shannon)

    bins = linspace(0, 850, 100)
    axes = plt.gca()
    axes.set_ylim([0, 30*m + 3])
    axes.set_xlim([-2, 850])
    br_fixed = [int(i) for i in br_fixed]
    br_flex = [int(i) for i in br_flex]
    br_shannon = [int(i) for i in br_shannon]
    plt.hist(br_fixed, bins, label='fixed')
    plt.hist(br_flex, bins, label='flex')
    plt.hist(br_shannon, bins, label="shannon")
    plt.legend(loc="upper right")
    plt.title("Uniform traffix matrix allocation for different BR strategies, M = " + str(m))
    plt.savefig("question_7_" + str(m) + ".png")
    plt.show()


if __name__ == "__main__":
    main()
