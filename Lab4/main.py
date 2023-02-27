import network as nc
from random import sample, seed
import matplotlib.pyplot as plt
possible_nodes = ["A", "B", "C", "D", "E", "F", "G"]


def main():
    network = nc.Network("nodes.json", 10)
    network.connect()
    network.network_analysis()
    network.routing_space_update()
    print(network.get_route_space())



if __name__ == "__main__":
    main()
