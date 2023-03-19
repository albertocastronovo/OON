from network import Network
import network_tm as ntm
from network_elements import Connection
import matplotlib.pyplot as plt
from numpy import arange
from random import sample, seed
from copy import deepcopy
from statistics import mean

possible_nodes = ["A", "B", "C", "D", "E", "F", "G"]


def main():
    seed(22032023)
    con_fi = [Connection(sample(possible_nodes, 2), 1e-3) for _ in range(100)]
    con_fl = deepcopy(con_fi)
    con_sh = deepcopy(con_fi)

    n_fi = Network("network/not_full_network_fixed.json", 6)
    n_fl = Network("network/not_full_network_flex.json", 6)
    n_sh = Network("network/not_full_network_shannon.json", 6)
    n_fi.connect()
    n_fi.network_analysis()
    n_fi.routing_space_update()
    n_fl.connect()
    n_fl.network_analysis()
    n_fl.routing_space_update()
    n_sh.connect()
    n_sh.network_analysis()
    n_sh.routing_space_update()
    print("fixed stream")
    ct_fi = n_fi.stream(con_fi, best="snr")
    print("flex stream")
    ct_fl = n_fl.stream(con_fl, best="snr")
    print("shannon stream")
    ct_sh = n_sh.stream(con_sh, best="snr")

    snr_fi = [c.get_snr() for c in con_fi]
    snr_fl = [c.get_snr() for c in con_fl]
    snr_sh = [c.get_snr() for c in con_sh]

    br_fi = [c.get_bit_rate() for c in con_fi]
    br_fl = [c.get_bit_rate() for c in con_fl]
    br_sh = [c.get_bit_rate() for c in con_sh]

    bins_range = arange(20, 42.25, 0.25)

    plt.hist(x=snr_fi, alpha=0.33, bins=bins_range, color="red",
             label=f"fixed, {n_fi.get_route_space_occupancy() * 100.0:.3f}% free")
    plt.hist(x=snr_fl, alpha=0.33, bins=bins_range, color="blue",
             label=f"flex, {n_fl.get_route_space_occupancy() * 100.0:.3f}% free")
    plt.hist(x=snr_sh, alpha=0.33, bins=bins_range, color="green",
             label=f"shannon, {n_sh.get_route_space_occupancy() * 100.0:.3f}% free")
    plt.title(f"Dynamic SM with different transceivers")
    plt.xlabel("SNR [dB]")
    plt.ylabel("# of connections")
    plt.xlim([20, 42.25])
    plt.legend(loc="upper right")
    plt.show()

    bins_br = arange(0, max(br_sh) + 20, 5)

    br_fi_nz = [x for x in br_fi if x != 0]
    br_fl_nz = [x for x in br_fl if x != 0]
    br_sh_nz = [x for x in br_sh if x != 0]

    plt.yscale("log")
    plt.hist(x=br_fi, alpha=0.33, bins=bins_br, color="red",
             label=f"fixed, tot {sum(br_fi):.0f} ({ct_fi.get('allocated') * 100.0:.0f} allocated), avg {mean(br_fi_nz):.1f}")
    plt.hist(x=br_fl, alpha=0.33, bins=bins_br, color="blue",
             label=f"flex, tot {sum(br_fl):.0f} ({ct_fl.get('allocated') * 100.0:.0f} allocated), avg {mean(br_fl_nz):.1f}")
    plt.hist(x=br_sh, alpha=0.33, bins=bins_br, color="green",
             label=f"shannon, tot {sum(br_sh):.1f} ({ct_sh.get('allocated') * 100.0:.0f} allocated), avg {mean(br_sh_nz):.1f}")
    plt.title(f"Bit rates of connections with different transceivers")
    plt.xlabel("Bit rate [GBit/s]")
    plt.ylabel("# of connections")
    plt.ylim([0.1, 1000])
    plt.legend(loc="upper right")
    plt.show()

    ntm_fi = ntm.Network("network/not_full_network_fixed.json", 6)
    ntm_fl = ntm.Network("network/not_full_network_flex.json", 6)
    ntm_sh = ntm.Network("network/not_full_network_shannon.json", 6)
    ntm_fi.connect()
    ntm_fi.network_analysis()
    ntm_fi.routing_space_update()
    ntm_fl.connect()
    ntm_fl.network_analysis()
    ntm_fl.routing_space_update()
    ntm_sh.connect()
    ntm_sh.network_analysis()
    ntm_sh.routing_space_update()

    m_range = range(1, 16)
    fi_valid_cons = []
    fl_valid_cons = []
    sh_valid_cons = []
    fi_rejected_cons = []
    fl_rejected_cons = []
    sh_rejected_cons = []
    fi_snr = []
    fl_snr = []
    sh_snr = []
    fi_cap = []
    fl_cap = []
    sh_cap = []
    fi_congestion = []
    fl_congestion = []
    sh_congestion = []

    for m in m_range:
        print("m = " + str(m))
        ntm_fi.reset_all_nodes_sm()
        ntm_fi.reset_route_space()
        ntm_fi.reset_all_lines()

        ntm_fl.reset_all_nodes_sm()
        ntm_fl.reset_route_space()
        ntm_fl.reset_all_lines()

        ntm_sh.reset_all_nodes_sm()
        ntm_sh.reset_route_space()
        ntm_sh.reset_all_lines()

        utm_fi = {
            starting_node: {
                ending_node: m*100.0*(starting_node != ending_node)
                for ending_node in possible_nodes
            }
            for starting_node in possible_nodes
        }
        utm_fl = deepcopy(utm_fi)
        utm_sh = deepcopy(utm_fi)

        valid_cons = {"fi": 0, "fl": 0, "sh": 0}
        refused_cons = {"fi": 0, "fl": 0, "sh": 0}
        excess_cons = {"fi": 0, "fl": 0, "sh": 0}
        snr_sum = {"fi": 0, "fl": 0, "sh": 0}
        capacity = {"fi": 0, "fl": 0, "sh": 0}

        for _ in range(m*42):
            return_fi, con_fi = ntm_fi.stream_random_from_tm(utm_fi)
            return_fl, con_fl = ntm_fl.stream_random_from_tm(utm_fl)
            return_sh, con_sh = ntm_sh.stream_random_from_tm(utm_sh)

            snr_sum["fi"] += con_fi.get_snr()
            snr_sum["fl"] += con_fl.get_snr()
            snr_sum["sh"] += con_sh.get_snr()

            if return_fi == 0:
                valid_cons["fi"] += 1
                capacity["fi"] += con_fi.get_bit_rate()
            elif return_fi == -1:
                refused_cons["fi"] += 1
            else:
                excess_cons["fi"] += 1

            if return_fl == 0:
                valid_cons["fl"] += 1
                capacity["fl"] += con_fl.get_bit_rate()
            elif return_fl == -1:
                refused_cons["fl"] += 1
            else:
                excess_cons["fl"] += 1

            if return_sh == 0:
                valid_cons["sh"] += 1
                capacity["sh"] += con_sh.get_bit_rate()
            elif return_sh == -1:
                refused_cons["sh"] += 1
            else:
                excess_cons["sh"] += 1

        fi_valid_cons.append(valid_cons["fi"])
        fl_valid_cons.append(valid_cons["fl"])
        sh_valid_cons.append(valid_cons["sh"])
        fi_rejected_cons.append(refused_cons["fi"])
        fl_rejected_cons.append(refused_cons["fl"])
        sh_rejected_cons.append(refused_cons["sh"])

        fi_snr.append(float(snr_sum["fi"]) / valid_cons["fi"])
        fl_snr.append(float(snr_sum["fl"]) / valid_cons["fl"])
        sh_snr.append(float(snr_sum["sh"]) / valid_cons["sh"])

        fi_cap.append(float(capacity["fi"]))
        fl_cap.append(float(capacity["fl"]))
        sh_cap.append(float(capacity["sh"]))

        fi_congestion.append(ntm_fi.get_route_space_occupancy())
        fl_congestion.append(ntm_fl.get_route_space_occupancy())
        sh_congestion.append(ntm_sh.get_route_space_occupancy())

    plt.plot(list(m_range), fi_valid_cons, alpha=0.9, color="red", label="allocated, fixed")
    plt.plot(list(m_range), fl_valid_cons, alpha=0.9, color="green", label="allocated, flex")
    plt.plot(list(m_range), sh_valid_cons, alpha=0.9, color="blue", label="allocated, shannon")
    plt.plot(list(m_range), fi_rejected_cons, alpha=0.33, color="red", label="rejected, fixed")
    plt.plot(list(m_range), fl_rejected_cons, alpha=0.33, color="green", label="rejected, flex")
    plt.plot(list(m_range), sh_rejected_cons, alpha=0.33, color="blue", label="rejected, shannon")
    plt.legend(loc="upper left")
    plt.title("valid connections vs transceiver and matrix")
    plt.xlabel("Order of traffic matrix, M")
    plt.ylabel("# of connections")
    plt.show()

    plt.plot(list(m_range), fi_snr, alpha=0.33, color="red",  label="fixed")
    plt.plot(list(m_range), fl_snr, alpha=0.33, color="green", label="flex")
    plt.plot(list(m_range), sh_snr, alpha=0.33, color="blue", label="shannon")
    plt.legend(loc="lower left")
    plt.title("average GSNR vs transceiver and matrix")
    plt.xlabel("Order of traffic matrix, M")
    plt.ylabel("GSNR [dB]")
    plt.show()

    requested_capacity = [m*100*42 for m in m_range]

    plt.plot(list(m_range), fi_cap, alpha=0.33, color="red",  label="fixed")
    plt.plot(list(m_range), fl_cap, alpha=0.33, color="green", label="flex")
    plt.plot(list(m_range), sh_cap, alpha=0.33, color="blue", label="shannon")
    plt.plot(list(m_range), requested_capacity, alpha = 0.3, color="black", label="requested")
    plt.legend(loc="upper left")
    plt.title("total capacity vs transceiver and matrix")
    plt.xlabel("Order of traffic matrix, M")
    plt.ylabel("Capacity [Gbps]")
    plt.show()

    plt.plot(list(m_range), [(1-x)*100 for x in fi_congestion], alpha=0.33, color="red",  label="fixed")
    plt.plot(list(m_range), [(1-x)*100 for x in fl_congestion], alpha=0.33, color="green", label="flex")
    plt.plot(list(m_range), [(1-x)*100 for x in sh_congestion], alpha=0.33, color="blue", label="shannon")
    plt.legend(loc="lower right")
    plt.title("Spectral occupancy vs transceiver and matrix")
    plt.xlabel("Order of traffic matrix, M")
    plt.ylabel("Unavailable paths [%]")
    plt.show()


if __name__ == "__main__":
    main()
