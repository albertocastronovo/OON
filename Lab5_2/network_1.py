from json import load
from math import sqrt, log10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network_elements import Node, Line, Connection, Lightpath


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
        self.__rsu_vectorized = np.vectorize(self.__rsu_to_vectorize)

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

        return output_nodes, output_lines

    @staticmethod
    def lines_in_path(path: list[str]):
        return [path[i] + path[i+1] for i in range(len(path)-1)]

    # class methods

    def connect(self):
        for node in self.__nodes.values():  # set successive attribute of each node
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
        self.__nodes[starting_node].propagate(signal)

    def probe(self, signal: Lightpath):
        starting_node = signal.get_path()[0]
        self.__nodes[starting_node].probe(signal)

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
                ["->".join(path)] + ["1" for i in range(self.__channels+1)]
            )
        self.__weighted_paths = pd.DataFrame(
            paths_for_wp,
            columns=["Path", "Latency [s]", "Noise Power [W]", "SNR [dB]"]
        )
        self.__route_space = pd.DataFrame(
            paths_for_rs,
            columns=["Path", "CH_ANY"] + ["CH_" + str(i+1) for i in range(self.__channels)]
        )
        for col in self.__route_space.columns:
            if col != "Path":
                self.__route_space[col].values[:] = 1

    def find_best_snr(self, input_node: str, output_node: str) -> str:
        paths_subset = self.__route_space.index[
            (self.__route_space["Path"].str.startswith(input_node)) &
            (self.__route_space["Path"].str.endswith(output_node)) &
            (self.__route_space["CH_ANY"] == 1)
        ].tolist()
        if len(paths_subset) == 0:
            return "empty"
        max_index = self.__weighted_paths.iloc[paths_subset]["SNR [dB]"].idxmax(axis=0)
        return str(self.__weighted_paths.iloc[max_index]["Path"])

    def find_best_latency(self, input_node: str, output_node: str) -> str:
        paths_subset = self.__route_space.index[
            (self.__route_space["Path"].str.startswith(input_node)) &
            (self.__route_space["Path"].str.endswith(output_node)) &
            (self.__route_space["CH_ANY"] == 1)
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

    def occupy_lines_from_path(self, path: list[str]):
        lines = self.lines_in_path(path)
        for line in lines:
            self.__lines[line].set_state(0)

    def occupy_all_subpaths(self, path: str, channel: int):
        self.__route_space.loc[self.__route_space["Path"].str.contains(path), "CH_" + str(channel)] = 0
        self.route_space_update_any()
        # if channel == self.__channels:
        # self.__route_space.loc[self.__route_space["Path"].str.contains(path), "CH_ANY"] = 0

    def route_space_update_any(self):
        ch_columns = [c for c in list(self.__route_space.columns) if c != "CH_ANY" and c != "Path"]
        self.__route_space["CH_ANY"] = self.__route_space[ch_columns].any(axis=1).astype(int)

    def get_route_space_occupancy(self):
        ch_columns = [c for c in list(self.__route_space.columns) if c != "CH_ANY" and c != "Path"]
        return self.__route_space[ch_columns].sum(axis=0).sum() / (float(len(self.__route_space.index))*self.__channels)

    def reset_route_space(self):
        self.__route_space.replace(to_replace=0, value=1, inplace=True)

    def get_first_free_channel(self, path: str) -> int:
        for i in range(1, self.__channels + 1):
            if self.__route_space.loc[self.__route_space["Path"] == path, "CH_" + str(i)].tolist()[0]:
                return i
        return -1

    def stream(self, connection_list: list[Connection], best="latency") -> float:
        all_connections = len(connection_list)
        success = 0
        for connection in connection_list:
            if best == "latency":
                path = self.find_best_latency(connection.get_input(), connection.get_output()).split("->")
            elif best == "snr":
                path = self.find_best_snr(connection.get_input(), connection.get_output()).split("->")
            else:
                return -1
            if path[0] == "empty":
                connection.set_latency(-1)
                connection.set_snr(0.0)
            else:
                success += 1
                channel = self.get_first_free_channel("->".join(path))
                # print("connessione accettata: " + self.__route_space.loc[self.__route_space.Path == "->".join(path)].to_string())
                test_signal = Lightpath(signal_power_value=connection.get_signal_power(),
                                        given_path=path,
                                        selected_channel=channel
                                        )
                self.propagate(test_signal)
                #self.occupy_all_subpaths("->".join(path), channel)
                self.routing_space_update()
                connection.set_latency(test_signal.get_latency())
                connection.set_snr(test_signal.get_snr())
        return float(success)/float(all_connections)

    def __rsu_to_vectorize(self, path, channel):
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
        for i in range(1, self.__channels + 1):
            self.__route_space["CH_" + str(i)] = pd.Series(
                self.__rsu_vectorized(
                    self.__route_space.Path,
                    i
                )
            )
        self.route_space_update_any()
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

    # class overloads

    def __repr__(self):
        return "Network object"

    def __str__(self):
        message = f"Network with {len(self.__nodes)} nodes and {len(self.__lines)} lines\n"
        for node in self.__nodes.values():
            message += str(node)
        for line in self.__lines.values():
            message += str(line)
        return message
