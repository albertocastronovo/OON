import network_1 as nc1
import network_3 as nc3
import network_t as nct
import network_elements_t as ne
from random import sample, seed
import matplotlib.pyplot as plt
from copy import deepcopy
from textwrap import dedent
from statistics import mean
from numpy import arange

possible_nodes = ["A", "B", "C", "D", "E", "F", "G"]


def main():
    seed(22032023)
    n1fcon = [ne.Connection(sample(possible_nodes, 2), 1e-3) for _ in range(100)]
    n1ncon = deepcopy(n1fcon)
    n3fcon = deepcopy(n1fcon)
    n3ncon = deepcopy(n1fcon)
    ntficon = deepcopy(n1fcon)
    ntflcon = deepcopy(n1fcon)
    ntshcon = deepcopy(n1fcon)

    n1f = nc1.Network(file_path="network/full_network.json", channels=6)
    n1n = nc1.Network(file_path="network/not_full_network.json", channels=6)

    n3f = nc3.Network(file_path="network/full_network.json", channels=6)
    n3n = nc3.Network(file_path="network/not_full_network.json", channels=6)

    ntfi = nct.Network(file_path="network/not_full_network_fixed.json", channels=6)
    ntfl = nct.Network(file_path="network/not_full_network_flex.json", channels=6)
    ntsh = nct.Network(file_path="network/not_full_network_shannon.json", channels=6)

    n1f.connect()
    n1f.network_analysis()
    n1f.routing_space_update()
    n1n.connect()
    n1n.network_analysis()
    n1n.routing_space_update()
    n3f.connect()
    n3f.network_analysis()
    n3f.routing_space_update()
    n3n.connect()
    n3n.network_analysis()
    n3n.routing_space_update()
    ntfi.connect()
    ntfi.network_analysis()
    ntfi.routing_space_update()
    ntfl.connect()
    ntfl.network_analysis()
    ntfl.routing_space_update()
    ntsh.connect()
    ntsh.network_analysis()
    ntsh.routing_space_update()

    c1f = n1f.stream(n1fcon, best="snr")
    c1n = n1n.stream(n1ncon, best="snr")
    c3f = n3f.stream(n3fcon, best="snr")
    c3n = n3n.stream(n3ncon, best="snr")
    ctfi = ntfi.stream(ntficon, best="snr")
    ctfl = ntfl.stream(ntflcon, best="snr")
    ctsh = ntsh.stream(ntshcon, best="snr")

    snr1f = [c.get_snr() for c in n1fcon]
    snr1n = [c.get_snr() for c in n1ncon]
    snr3f = [c.get_snr() for c in n3fcon]
    snr3n = [c.get_snr() for c in n3ncon]
    snrtfi = [c.get_snr() for c in ntficon]
    snrtfl = [c.get_snr() for c in ntflcon]
    snrtsh = [c.get_snr() for c in ntshcon]
    brfi = [c.get_bit_rate() for c in ntficon]
    brfl = [c.get_bit_rate() for c in ntflcon]
    brsh = [c.get_bit_rate() for c in ntshcon]

    #   plot of snr for static matrix networks

    bins_range = arange(0, 42.25, 0.25)
    plt.hist(x=snr1f, alpha=0.5, bins=bins_range, color="red", label=f"full, {n1f.get_route_space_occupancy()*100.0:.3f}% free")
    plt.hist(x=snr1n, alpha=0.5, bins=bins_range, color="blue", label=f"not full, {n1n.get_route_space_occupancy()*100.0:.3f}% free")
    plt.title(f"Static switching matrix, average {mean(snr1f):.2f} dB")
    plt.xlabel("SNR [dB]")
    plt.ylabel("# of connections")
    plt.legend(loc="upper left")
    plt.show()

    #   plot of snr for dynamic matrix networks

    plt.hist(x=snr3f, alpha=0.5, bins=bins_range, color="red", label=f"full, {n3f.get_route_space_occupancy()*100.0:.3f}% free")
    plt.hist(x=snr3n, alpha=0.5, bins=bins_range, color="blue", label=f"not full, {n3n.get_route_space_occupancy()*100.0:.3f}% free")
    plt.title(f"Dynamic switching matrix, average {mean(snr3f):.2f} dB")
    plt.xlabel("SNR [dB]")
    plt.ylabel("# of connections")
    plt.legend(loc="upper left")
    plt.show()

    # plot of difference between static and dynamic matrices (not full case)

    plt.hist(x=snr1n, alpha=0.5, bins=bins_range, color="green",
             label=f"static, {n1n.get_route_space_occupancy() * 100.0:.3f}% free")
    plt.hist(x=snr3n, alpha=0.5, bins=bins_range, color="magenta",
             label=f"dynamic, {n3n.get_route_space_occupancy() * 100.0:.3f}% free")
    plt.title("SNR for static vs dynamic not full switching matrix")
    plt.xlabel("SNR [dB]")
    plt.ylabel("# of connections")
    plt.legend(loc="upper right")
    plt.xlim([25, 42])
    plt.show()

    #   plot of bit rates for transceiver networks

    bins_br = arange(0, max(brsh) + 20, 5)
    plt.yscale("log")
    plt.hist(x=brfi, alpha=0.33, bins=bins_br, color="red",
             label=f"fixed, tot {sum(brfi):.0f} ({ctfi.get('allocated')*100.0:.0f} allocated), avg {mean(brfi):.1f}")
    plt.hist(x=brfl, alpha=0.33, bins=bins_br, color="blue",
             label=f"flex, tot {sum(brfl):.0f} ({ctfl.get('allocated')*100.0:.0f} allocated), avg {mean(brfl):.1f}")
    plt.hist(x=brsh, alpha=0.33, bins=bins_br, color="green",
             label=f"shannon, tot {sum(brsh):.1f} ({ctsh.get('allocated')*100.0:.0f} allocated), avg {mean(brsh):.1f}")
    plt.title(f"Bit rates of connections with different transceivers")
    plt.xlabel("Bit rate [GBit/s]")
    plt.ylabel("# of connections")
    plt.ylim([0.1, 1000])
    plt.legend(loc="upper right")
    plt.show()

    #   plot of snr for transceiver networks

    plt.hist(x=snrtfi, alpha=0.33, bins=bins_range, color="red",
             label=f"fixed, {ntfi.get_route_space_occupancy() * 100.0:.3f}% free")
    plt.hist(x=snrtfl, alpha=0.33, bins=bins_range, color="blue",
             label=f"flex, {ntfl.get_route_space_occupancy() * 100.0:.3f}% free")
    plt.hist(x=snrtsh, alpha=0.33, bins=bins_range, color="green",
             label=f"shannon, {ntsh.get_route_space_occupancy() * 100.0:.3f}% free")
    plt.title(f"Dynamic SM with different transceivers")
    plt.xlabel("SNR [dB]")
    plt.ylabel("# of connections")
    plt.legend(loc="upper left")
    plt.show()

    #   print of info about how many connections have paths composed of 2, 3 or more nodes

    print(dedent(f"""\
        Number of 2-nodes connections:
        Dynamic matrix, full: {c3f.get("two-nodes")}
        Dynamic matrix, not full: {c3n.get("two-nodes")}
        Transceiver, fixed: {ctfi.get("two-nodes")}
        Transceiver, flex: {ctfl.get("two-nodes")}
        Transceiver, shannon: {ctsh.get("two-nodes")}
        
        Number of 3-nodes connections:
        Dynamic matrix, full: {c3f.get("three-nodes")}
        Dynamic matrix, not full: {c3n.get("three-nodes")}
        Transceiver, fixed: {ctfi.get("three-nodes")}
        Transceiver, flex: {ctfl.get("three-nodes")}
        Transceiver, shannon: {ctsh.get("three-nodes")}
    """))


def test():
    network = nct.Network("./network/full_network.json", 6)
    n1fcon = [ne.Connection(sample(possible_nodes, 2), 1e-3) for _ in range(10)]
    network.connect()
    network.network_analysis()
    network.routing_space_update()
    network.draw()
    network.stream(n1fcon, best="snr")


if __name__ == "__main__":
    main()
