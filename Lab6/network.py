from json import load
from math import sqrt, log10, log2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network_elements import Node, Line, Connection, Lightpath
from scipy.special import erfcinv


class Network:
    def __init__(self, file_path: str = None, channels: int = 10):
        if file_path is None:
            self.__nodes: dict[str: Node] = {}
            self.__lines: dict[str: Line] = {}
        else:
            self.__nodes, self.__lines = self.parse_json_to_elements(file_path)
        self.__all_paths: list[list[str]] = []
        self.__weighted_paths: pd.DataFrame = pd.DataFrame()
        self.__route_space: pd.DataFrame = pd.DataFrame() # path vs channel availability
        self.__channels: int = channels
        self.__capacity: dict[str: int] = {}

    @staticmethod
    def parse_json_to_elements(json_path: str) -> tuple[dict[str: Node], dict[str: Line]]:
        output_nodes = {}
        output_lines = {}
        with open(json_path, "r") as json_file:
            json_data = load(json_file)
            for key in sorted(json_data.keys()):
                json_data[key]["label"] = key
                output_nodes[key] = Node(json_data[key])
                for node in json_data[key]["connected_nodes"]:
                    line_label = key + node
                    line_length = sqrt(
                        (json_data[key]["position"][0] - json_data[node]["position"][0])**2 +
                        (json_data[key]["position"][1] - json_data[node]["position"][1])**2
                    )
                    output_lines[line_label] = Line(line_label, line_length)
                    # print("Line " + line_label + " has length " + str(line_length) + " m")

        return output_nodes, output_lines

    @staticmethod
    def lines_in_path(path: list[str]):
        return [path[i] + path[i+1] for i in range(len(path)-1)]

    # class methods

    def connect(self):
        for node in self.__nodes.values():  # set successive attribute of each node
            """
            node.set_switching_matrix(
                {
                    n: {
                        m: np.array([
                            int(n != m) for i in range(self.__channels)
                        ]) for m in self.__nodes.keys()
                    } for n in self.__nodes.keys()
                }
            )
            """
            node.set_successive(
                {
                    node.get_label() + connected_node: self.__lines[node.get_label() + connected_node]
                    for connected_node in node.get_connected_nodes()
                }
            )
        for line in self.__lines.values():  # set successive attribute of each line
            line.set_successive(
                {
                    label_char: self.__nodes[label_char]
                    for label_char in line.get_label()
                }
            )
            line.set_state([1 for i in range(self.__channels)])
        self.__all_paths = list(self.find_all_paths())

    def find_all_paths(self, path: list = None, min_length: int = 2) -> list[list[str]]:
        if path is None:
            for node in self.__nodes.keys():
                yield from self.find_all_paths(path=[node], min_length=2)
        else:
            if len(path) >= min_length:
                yield path
            if path[-1] in path[:-1]:
                return
            current_node = path[-1]
            for next_node in self.__nodes[current_node].get_connected_nodes():
                if next_node not in path:
                    yield from self.find_all_paths(path=path + [next_node], min_length=2)

    def find_paths(self, starting_node: str, ending_node: str) -> list[list[str]]:
        return [path for path in self.__all_paths if path[0] == starting_node and path[-1] == ending_node]

    def is_path_free(self, path: list[str]):
        is_free = 1
        for label in self.lines_in_path(path):
            is_free *= self.__lines[label].get_state()
        return is_free

    def propagate(self, signal: Lightpath):
        starting_node = signal.get_path()[0]
        self.__nodes[starting_node].propagate(signal, self.__channels)

    def probe(self, signal: Lightpath):
        starting_node = signal.get_path()[0]
        self.__nodes[starting_node].probe(signal, self.__channels)

    def draw(self):
        # draw nodes
        for label in self.__nodes.keys():
            node_pos = self.__nodes[label].get_position()
            plt.scatter(*node_pos, s=500, marker='o', color='r', linewidths=0)
            plt.annotate(label, xy=node_pos, xytext=(-3, -6), textcoords="offset pixels")
        # draw lines

        for line in self.__lines.keys():
            node_pos = [self.__nodes[line[0]].get_position(), self.__nodes[line[1]].get_position()]
            plt.plot(*zip(*node_pos), color='b', zorder=-1)

        # plot parameters
        plt.axis("off")
        plt.title("Network")
        plt.show()

    def network_analysis(self):
        paths_for_wp = []
        paths_for_rs = []
        for path in self.__all_paths:
            test_signal = Lightpath(signal_power_value=1e-3)
            test_signal.set_path(path)
            self.probe(test_signal)
            snr = 10*log10(test_signal.get_signal_power()/test_signal.get_noise_power())
            paths_for_wp.append(
                    [
                        "->".join(path),
                        test_signal.get_latency(),
                        test_signal.get_noise_power(),
                        snr
                    ]
                )
            paths_for_rs.append(
                ["->".join(path)] + [str(self.__channels)] + ["1" for i in range(self.__channels)]
            )
        self.__weighted_paths = pd.DataFrame(
            paths_for_wp,
            columns=["Path", "Latency [s]", "Noise Power [W]", "SNR [dB]"]
        )
        self.__route_space = pd.DataFrame(
            paths_for_rs,
            columns=["Path", "CH_CNT"] + ["CH_" + str(i+1) for i in range(self.__channels)]
        )
        for col in self.__route_space.columns:
            if col != "Path":
                self.__route_space[col].values[:] = 1
        self.routing_space_update()

    def find_best_snr(self, input_node: str, output_node: str) -> str:
        paths_subset = self.__route_space.index[
            (self.__route_space["Path"].str.startswith(input_node)) &
            (self.__route_space["Path"].str.endswith(output_node)) &
            (self.__route_space["CH_CNT"] > 0)
        ].tolist()
        if len(paths_subset) == 0:
            return "empty"
        max_index = self.__weighted_paths.iloc[paths_subset]["SNR [dB]"].idxmax(axis=0)
        return str(self.__weighted_paths.iloc[max_index]["Path"])

    def find_best_latency(self, input_node: str, output_node: str) -> str:
        paths_subset = self.__route_space.index[
            (self.__route_space["Path"].str.startswith(input_node)) &
            (self.__route_space["Path"].str.endswith(output_node)) &
            (self.__route_space["CH_CNT"] > 0)
        ].tolist()
        if len(paths_subset) == 0:
            return "empty"
        min_index = self.__weighted_paths.iloc[paths_subset]["Latency [s]"].idxmin(axis=0)
        return str(self.__weighted_paths.iloc[min_index]["Path"])

    def set_path_state(self, path: list[str], new_state: int, channel: int):
        self.__weighted_paths.loc[
            self.__weighted_paths["Path"] == "->".join(path),
            "CH_" + str(channel)
        ] = int(new_state)

    def set_all_paths_state(self, new_state: int, channel: int):
        self.__weighted_paths["CH_" + str(channel)] = new_state

    def get_first_free_channel(self, path: str) -> int:
        for i in range(1, self.__channels + 1):
            if self.__route_space.loc[self.__route_space["Path"] == path, "CH_" + str(i)].tolist()[0]:
                return i
        return -1

    def stream(self, connection_list: list[Connection], best="latency"):
        total_count = 0
        fail_count = 0
        success_count = 0

        for connection in connection_list:
            total_count += 1
            if best == "latency":
                path = self.find_best_latency(connection.get_input(), connection.get_output()).split("->")
            elif best == "snr":
                path = self.find_best_snr(connection.get_input(), connection.get_output()).split("->")
            else:
                return
            try:
                strategy = self.__nodes[path[0]].get_transceiver()
            except KeyError:
                strategy = "fixed_rate"
            bit_rate = self.calculate_bit_rate(path="->".join(path), strategy=strategy)
            # print("connection #" + str(total_count) + " bit rate: " + str(bit_rate))

            if path[0] == "empty" or bit_rate is None or bit_rate < 0.1:
                fail_count += 1
                connection.set_latency(-1)
                connection.set_snr(0.0)
            else:
                success_count += 1
                connection.set_bit_rate(bit_rate)
                channel = self.get_first_free_channel("->".join(path))
                test_signal = Lightpath(signal_power_value=connection.get_signal_power(),
                                        given_path=path.copy(),
                                        selected_channel=channel
                                        )
                self.propagate(test_signal)
                # switching matrix occupation
                self.occupy_switching_matrices(path, channel)
                connection.set_latency(test_signal.get_latency())
                connection.set_snr(test_signal.get_snr())
                print("BR: " + str(bit_rate))
                self.routing_space_update()
        print("total connections: " + str(total_count))
        print("total failures: " + str(fail_count))
        self.set_capacity(
            {
                "total": total_count,
                "accepted": success_count,
                "rejected": fail_count
            }
        )
        self.restore_all_matrices()

    def restore_all_matrices(self):
        for node in self.__nodes.values():
            node.restore_switching_matrix()

    def occupy_switching_matrices(self, path: list[str], channel: int):
        if len(path) > 2:
            for i in range(1, len(path)-1):
                self.__nodes[path[i]].occupy_switching_matrix(path[i-1], path[i+1], channel)

    def get_all_lines_state(self):
        message = ""
        for key, value in self.__lines.items():
            message += key + ": " + ", ".join([str(v) for v in value.get_state()]) + "\n"
        return message

    def __to_vectorize(self, path, channel):
        final_value = 1
        path_list = path.split("->")
        lines = self.lines_in_path(path_list)
        for line in lines:
            final_value *= self.__lines[line].get_channel_state(channel)
        if len(path_list) > 2:
            for n in range(1, len(path_list)-1):
                final_value *= self.__nodes[path_list[n]].evaluate_switching_matrix(path_list[n-1],
                                                                                    path_list[n+1],
                                                                                    channel
                                                                                    )
        return final_value

    def routing_space_update(self):
        v = np.vectorize(self.__to_vectorize)
        for i in range(1, self.__channels + 1):
            self.__route_space["CH_" + str(i)] = pd.Series(v(self.__route_space.Path, i))
        self.__route_space["CH_CNT"] = self.__route_space[list(self.__route_space.columns)[2:]].sum(axis=1)

    def get_total_free_channels(self):
        return self.__route_space["CH_CNT"].sum()

    def calculate_bit_rate(self, path: str, strategy: str = "fixed_rate"):
        # print("path in calcuate br: " + path)
        try:
            gsnr = float(self.__weighted_paths.loc[self.__weighted_paths["Path"] == path, "SNR [dB]"])
        except:
            return None
        # print("gsnr: " + str(gsnr))
        r_s = 32.0      # Rs = symbol rate, fixed to 32 (GHz)
        b_n = 12.5      # Bn = noise bandwidth, fixed to 12.5 (GHz)
        ber_t = 1e-3    # BERt = bit error rate, fixed to 10^-3

        if strategy == "fixed_rate":
            min_gsnr = 10.0*log10(2*(erfcinv(2*ber_t)**2)*r_s/b_n)
            # print("fixed rate min gsnr: " + str(min_gsnr))
            if gsnr >= min_gsnr:
                return 100
            else:
                return 0
        elif strategy == "flex_rate":
            gsnr_2 = 10.0*log10(2.0*(erfcinv(2.0*ber_t)**2)*r_s/b_n)
            # print("flex rate min gsnr: " + str(gsnr_2))
            gsnr_143 = 10.0*log10(14.0/3.0 * (erfcinv(3.0*ber_t/2.0)**2)*r_s/b_n)
            # print("flex rate min gsnr: " + str(gsnr_143))
            gsnr_10 = 10.0*log10(10.0 * (erfcinv(8.0*ber_t/3.0)**2)*r_s/b_n)
            # print("flex rate min gsnr: " + str(gsnr_10))
            if gsnr < gsnr_2:
                return 0
            elif gsnr_2 <= gsnr < gsnr_143:
                return 100
            elif gsnr_143 <= gsnr < gsnr_10:
                return 200
            else:
                return 400
        elif strategy == "shannon":
            linear_gsnr = 10**(gsnr/10.0)
            return 2.0*r_s*log2(1 + linear_gsnr*r_s/b_n)
        else:
            return 0

    # class getters

    def get_nodes(self) -> dict[str: Node]:
        return self.__nodes

    def get_lines(self) -> dict[str: Line]:
        return self.__lines

    def get_all_paths(self) -> list[list[str]]:
        return self.__all_paths

    def get_weighted_paths(self) -> pd.DataFrame:
        return self.__weighted_paths

    def get_route_space(self) -> pd.DataFrame:
        return self.__route_space

    def get_capacity(self) -> dict:
        return self.__capacity

    # class setters

    def set_nodes(self, new_nodes):
        try:
            self.__nodes = dict(new_nodes)
        except ValueError:
            print("The value given to the set_nodes method was not a valid dictionary.")

    def set_lines(self, new_lines):
        try:
            self.__lines = dict(new_lines)
        except ValueError:
            print("The value given to the set_lines method was not a valid dictionary.")

    def set_all_paths(self, new_all_paths):
        try:
            self.__all_paths = list(new_all_paths)
        except ValueError:
            print("The value given to the set_all_paths method was not a valid list.")

    def set_weighted_paths(self, new_weighted_paths):
        try:
            self.__weighted_paths = pd.DataFrame(new_weighted_paths)
        except ValueError:
            print("The value given to the set_weighted_paths method could not be used to create a dataframe.")

    def set_capacity(self, new_capacity):
        try:
            self.__capacity = dict(new_capacity)
        except ValueError:
            print("The value given to the get_capacity method was not a valid dictionary.")

    # class overloads

    def __repr__(self):
        return "Network object"

    def __str__(self):
        message = "Network with " + str(len(self.__nodes)) + " nodes and " + str(len(self.__lines)) + " lines\n"
        for node in self.__nodes.values():
            message += str(node)
        for line in self.__lines.values():
            message += str(line)
        message += "\nWeighted paths:\n" + str(self.__weighted_paths)
        return message
