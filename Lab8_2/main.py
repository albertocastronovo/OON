from network import Network
from random import seed
import matplotlib.pyplot as plt
import time
from statistics import mean
from textwrap import dedent

#   Monte Carlo emulator(s)
#   The most relevant metrics are:
#   -   total capacity (the cumulative communication speed of all allocated connections)
#   -   per-link average capacity (the average communication speed per connection)
#   -   per-link GSNR (the average GSNR per connection)
#   -   min/max capacity (the minimum/maximum communication speed in the network)
#   -   min/max GSNR (the minimum/maximum GSNR among allocated connections)
#   -   blocking event count (the number of connections that were rejected due to the unavailability
#           of a suitable path or due to not meeting the GSNR requirements for the minimum com. speed)


def monte_carlo_congestion():
    #   provides the evolution of the metrics given the increment of M
    seed(22032023)
    n_channels = 6
    network = Network("network/not_full_network_flex.json", n_channels)
    network.connect()
    network.network_analysis()
    network.routing_space_update()
    possible_nodes = network.get_all_node_labels()

    total_capacity = []
    per_link_capacity = []
    per_link_gsnr = []
    min_capacity = []
    max_capacity = []
    min_gsnr = []
    max_gsnr = []
    block_count = []
    accepted_reqs = []
    requested_capacity = []
    attempted_connections = []

    m_range = range(1, 16)

    for m in m_range:
        print(f"processing m = {m}")
        network.reset_all_nodes_sm()
        network.reset_all_lines()
        network.reset_route_space()
        network.routing_space_update()

        traffic_matrix = {
            starting_node: {
                ending_node: m * 100.0 * (starting_node != ending_node)
                for ending_node in possible_nodes
            }
            for starting_node in possible_nodes
        }

        m_total_capacity = 0
        m_requested_capacity = m*100*(len(possible_nodes)**2)
        m_per_link_gsnr = 0
        m_min_capacity = 1000
        m_max_capacity = 0
        m_min_gsnr = 100
        m_max_gsnr = 0
        m_block_count = 0
        m_accepted_reqs = 0
        m_attempted_connections = m*n_channels*len(possible_nodes)

        for i in range(m_attempted_connections):
            return_value, streamed_con = network.stream_random_from_tm(traffic_matrix)
            if return_value == -2:      # the traffic matrix is empty
                break
            elif return_value == 0:     # accepted request
                m_accepted_reqs += 1
                m_total_capacity += streamed_con.get_bit_rate()
                m_per_link_gsnr += streamed_con.get_snr()
                m_min_capacity = min(m_min_capacity, streamed_con.get_bit_rate())
                m_max_capacity = max(m_max_capacity, streamed_con.get_bit_rate())
                m_min_gsnr = min(m_min_gsnr, streamed_con.get_snr())
                m_max_gsnr = max(m_max_gsnr, streamed_con.get_snr())

            else:                       # refused request
                m_block_count += 1
        try:
            m_per_link_capacity = float(m_total_capacity)/float(m_accepted_reqs)
            m_per_link_gsnr = float(m_per_link_gsnr)/float(m_accepted_reqs)
        except ZeroDivisionError:
            m_per_link_capacity = 0
            m_per_link_gsnr = 0
            m_min_capacity = 0
            m_max_capacity = 0
            m_min_gsnr = 0
            m_max_gsnr = 0

        total_capacity.append(m_total_capacity)
        per_link_capacity.append(m_per_link_capacity)
        per_link_gsnr.append(m_per_link_gsnr)
        min_capacity.append(m_min_capacity)
        max_capacity.append(m_max_capacity)
        min_gsnr.append(m_min_gsnr)
        max_gsnr.append(m_max_gsnr)
        block_count.append(m_block_count)
        accepted_reqs.append(m_accepted_reqs)
        requested_capacity.append(m_requested_capacity)
        attempted_connections.append(m_attempted_connections)

    # total capacity plot
    plt.plot(list(m_range), total_capacity, alpha=1.0, color="red", label="Total")
    plt.plot(list(m_range), requested_capacity, alpha=1.0, color="black", label="Requested from TM")
    plt.title(f"Requested and allocated total capacity vs. traffic matrix size")
    plt.ylabel("Capacity [Gbps]")
    plt.xlabel("Order of traffic matrix, M")
    plt.legend(loc="upper left")
    plt.show()

    # per link capacity plot
    plt.plot(list(m_range), per_link_capacity, alpha=1.0, color="green", label="Per-link average")
    plt.plot(list(m_range), min_capacity, alpha=1.0, color="blue", label="Per-link minimum")
    plt.plot(list(m_range), max_capacity, alpha=1.0, color="red", label="Per-link maximum")
    plt.title(f"Per-link avg, min and max capacity vs. traffic matrix size")
    plt.ylabel("Capacity [Gbps]")
    plt.xlabel("Order of traffic matrix, M")
    plt.legend(loc="upper left")
    plt.show()

    # GSNR plot
    plt.plot(list(m_range), per_link_gsnr, alpha=1.0, color="green", label="Per-link average")
    plt.plot(list(m_range), min_gsnr, alpha=1.0, color="blue", label="Per-link minimum")
    plt.plot(list(m_range), max_gsnr, alpha=1.0, color="purple", label="Per-link maximum")
    plt.title(f"Per-link avg, min and max GSNR vs. traffic matrix size")
    plt.ylabel("GSNR [dB]")
    plt.xlabel("Order of traffic matrix, M")
    plt.legend(loc="upper right")
    plt.show()

    # accepted and blocked connections plot
    plt.plot(list(m_range), block_count, alpha=1.0, color="green", label="Blocked")
    plt.plot(list(m_range), accepted_reqs, alpha=1.0, color="blue", label="Accepted")
    plt.plot(list(m_range), attempted_connections, alpha=1.0, color="black", label="Total")
    plt.title("Total count of accepted and blocked connections vs. traffic matrix size")
    plt.ylabel("# of connections")
    plt.xlabel("Order of traffic matrix, M")
    plt.legend(loc="upper left")
    plt.show()


def monte_carlo_single_tm(
        m: int = 1,
        strategy: str = "flex",
        channels: int = 6,
        moving_avg_size: int = 10,              # the window size of the moving average calculations
        allowed_block_fraction: float = 0.1,    # the % of connections that are allowed to be blocked before stopping
        snr_deviation: float = 0.15,            # the maximum allowed change in SNR to consider the dist. stable
        capacity_deviation: float = 0.3         # the maximum allowed change in capacity to consider the dist. stable
):
    #   with given M, provides the # of Monte Carlo runs for which
    #   the distributions of the metrics are stable.
    #   the metrics of interest are also provided.
    seed(22032023)
    network = Network(f"network/not_full_network_{strategy}.json", channels)
    network.connect()
    network.network_analysis()
    network.routing_space_update()
    possible_nodes = network.get_all_node_labels()

    traffic_matrix = {
        starting_node: {
            ending_node: m * 100.0 * (starting_node != ending_node)
            for ending_node in possible_nodes
        }
        for starting_node in possible_nodes
    }

    total_capacity = 0
    per_link_gsnr = []
    per_link_capacity = []
    block_count = 0
    accepted_reqs = 0
    requested_capacity = m*100*(len(possible_nodes)*(len(possible_nodes) - 1))
    total_connections = 0

    snr_moving_avg = []             # to evaluate the moving average of SNR
    capacity_moving_avg = []        # to evaluate the moving average of per-link capacity
    first_stable_snr_index = 0      # the n-th run for which SNR becomes stable for moving_avg_size connections
    first_stable_cap_index = 0      # the n-th run for which capacity becomes stable for moving_avg_size connections

    while 1:
        return_value, streamed_con = network.stream_random_from_tm(traffic_matrix)
        total_connections += 1
        if return_value == -2:  # stop if the traffic matrix is empty
            total_connections -= 1
            break
        elif return_value == -1:    # declined
            block_count += 1
            if block_count >= allowed_block_fraction*accepted_reqs:     # if there are too many declined connections
                break
        else:                       # accepted
            accepted_reqs += 1
            allocated_capacity = streamed_con.get_bit_rate()
            allocated_gsnr = streamed_con.get_snr()
            total_capacity += allocated_capacity
            per_link_capacity.append(allocated_capacity)
            per_link_gsnr.append(allocated_gsnr)

            if len(snr_moving_avg) >= moving_avg_size:
                if first_stable_snr_index == 0 and all(
                    abs(x - mean(snr_moving_avg)) <= snr_deviation*mean(snr_moving_avg)
                    for x in snr_moving_avg
                ):
                    first_stable_snr_index = total_connections - moving_avg_size
                del snr_moving_avg[0]
            snr_moving_avg.append(allocated_gsnr)

            if len(capacity_moving_avg) >= moving_avg_size:
                if first_stable_cap_index == 0 and all(
                    abs(x - mean(capacity_moving_avg)) <= capacity_deviation*mean(capacity_moving_avg)
                    for x in capacity_moving_avg
                ):
                    first_stable_cap_index = total_connections - moving_avg_size
                del capacity_moving_avg[0]
            capacity_moving_avg.append(allocated_capacity)

    min_capacity = min(per_link_capacity) if len(per_link_capacity) > 0 else 0
    max_capacity = max(per_link_capacity) if len(per_link_capacity) > 0 else 0
    min_gsnr = min(per_link_gsnr) if len(per_link_gsnr) > 0 else 0
    max_gsnr = max(per_link_gsnr) if len(per_link_gsnr) > 0 else 0

    capacity_met = total_capacity >= requested_capacity
    stable_snr_str = f"{first_stable_snr_index} connections" if first_stable_snr_index > 0\
        else f"never stable within {snr_deviation} of average"
    stable_cap_str = f"{first_stable_cap_index} connections" if first_stable_cap_index > 0\
        else f"never stable within {capacity_deviation} of average"

    average_snr = mean(per_link_gsnr) if len(per_link_gsnr) > 0 else 0
    average_cap = mean(per_link_capacity) if len(per_link_capacity) > 0 else 0

    print(dedent(f"""\
        Total capacity:             {total_capacity:.2f} Gbit/s
        Requested capacity:         {requested_capacity:.2f} Gbit/s
        Capacity requirement met:   {capacity_met}
        Per-link average capacity:  {average_cap:.2f} Gbit/s
        Per-link min capacity:      {min_capacity:.2f} Gbit/s
        Per-link max capacity:      {max_capacity:.2f} Gbit/s
        Per-link average GSNR:      {average_snr:.2f} dB
        Per-link min GSNR:          {min_gsnr:.2f} dB
        Per-link max GSNR:          {max_gsnr:.2f} dB
        Total runs:                 {total_connections}
        Allocated connections:      {accepted_reqs}
        Blocking event count:       {block_count}
        Terminated by blocks:       {block_count >= accepted_reqs*allowed_block_fraction}
        Capacity stable after:      {stable_cap_str}
        GSNR stable after:          {stable_snr_str}
    """))
    plt.plot(range(len(per_link_capacity)), per_link_capacity, color="red")
    plt.title(f"Per-link capacity for each allocated connection, m = {m}")
    plt.xlabel("Connections")
    plt.ylabel("Capacity [Gbps]")
    plt.ylim([0, max_capacity + 100])
    plt.show()

    plt.plot(range(len(per_link_gsnr)), per_link_gsnr, color="green")
    plt.title(f"Per-link GSNR for each allocated connection, m = {m}")
    plt.xlabel("Connections")
    plt.ylabel("GSNR [dB]")
    plt.ylim([0, max_gsnr + 5])
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    # monte_carlo_congestion()
    monte_carlo_single_tm(m=9, strategy="flex")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
