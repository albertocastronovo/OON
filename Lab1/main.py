import network_classes as nc
import matplotlib.pyplot as plt
from numpy import arange
from statistics import mean


def main():
    network = nc.Network("network.json")
    network.connect()
    network.network_analysis()
    network.draw()
    df = network.get_dataframe()
    snrs = df["SNR [dB]"]
    plt.hist(
        snrs,
        bins=arange(min(snrs), max(snrs) + 0.05, 0.05)
    )
    plt.title("Distribution of SNR across all paths")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Number of paths")
    plt.show()
    print(f"Number of paths: {len(snrs)}")
    print(f"Max SNR: {max(snrs):.2f} dB")
    print(f"Min SNR: {min(snrs):.2f} dB")
    print(f"Avg SNR: {mean(snrs):.2f} dB")

    lats = df["Latency [s]"]*1e03
    print(lats)
    plt.hist(
        lats,
        bins=arange(min(lats), max(lats) + 0.05, 0.05)
    )
    plt.title("Distribution of latency across all paths")
    plt.xlabel("Latency [ms]")
    plt.ylabel("Number of paths")
    plt.show()
    print(f"Max latency: {max(lats):.2f} ms")
    print(f"Min latency: {min(lats):.2f} ms")
    print(f"Avg latency: {mean(lats):.2f} ms")


if __name__ == "__main__":
    main()
