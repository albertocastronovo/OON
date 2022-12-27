import network_classes as nc


def main():
    network = nc.Network("nodes.json")
    print(network)
    network.connect()
    network.network_analysis()


if __name__ == "__main__":
    main()
