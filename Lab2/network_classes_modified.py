from json import load
from math import sqrt, log10
import matplotlib.pyplot as plt
import pandas as pd


#   Signal_information class
#   latency: total time delay due to the signal propagation
#       through any network element along the path.
#
class Signal_information:
    def __init__(self, signal_power_value=1e-3, given_path=None):
        self.__signal_power:   float = signal_power_value
        self.__noise_power:    float = 0.0
        self.__latency:        float = 0.0
        self.__path:           list[str] = []
        if given_path is not None:
            self.__path = given_path

    # class methods

    def update_signal_power(self, increment):
        try:
            self.__signal_power += float(increment)
        except ValueError:
            print("The value given to the update_signal_power method was not a float number.")

    def update_noise_power(self, increment):
        try:
            self.__noise_power += float(increment)
        except ValueError:
            print("The value given to the update_noise_power method was not a float number.")

    def update_latency(self, increment):
        try:
            self.__latency += float(increment)
        except ValueError:
            print("The value given to the update_latency method was not a float number.")

    def update_path(self, crossed_label):
        try:
            self.__path.remove(crossed_label)
        except ValueError:
            print("The label given to the update_path method was not in the path.")

    def get_snr(self) -> float:
        return 10*log10(self.__signal_power/self.__noise_power)

    # class getters

    def get_signal_power(self) -> float:
        return self.__signal_power

    def get_noise_power(self) -> float:
        return self.__noise_power

    def get_latency(self) -> float:
        return self.__latency

    def get_path(self) -> list[str]:
        return self.__path

    # class setters

    def set_signal_power(self, new_signal_power):
        try:
            self.__signal_power = float(new_signal_power)
        except ValueError:
            print("The value given to the set_signal_power method was not a float number.")

    def set_noise_power(self, new_noise_power):
        try:
            self.__noise_power = float(new_noise_power)
        except ValueError:
            print("The value given to the set_noise_power method was not a float number.")

    def set_latency(self, new_latency):
        try:
            self.__latency = float(new_latency)
        except ValueError:
            print("The value given to the set_latency method was not a float number.")

    def set_path(self, new_path):
        try:
            self.__path = list(new_path).copy()
        except ValueError:
            print("The value given to the set_latency method was not a float number.")

    # class overloads

    def __repr__(self):
        return "Signal_information object"

    def __str__(self):
        message = "Signal Information:\nSignal power: " + \
            "%.3f" % self.__signal_power + " W\nNoise power: " + \
            "%.3f" % self.__noise_power + " W\nLatency: " + \
            "%.3f" % self.__latency + " s\nPath: " + \
            ", ".join(self.__path) + "\n"
        return message


# Node class
#
class Node:
    def __init__(self, node_dictionary=None):
        try:
            self.__label:   str = node_dictionary["label"]
            self.__position: tuple[float, float] = tuple(node_dictionary["position"])
            self.__connected_nodes: list[str] = node_dictionary["connected_nodes"]
        except KeyError:
            print("Invalid node dictionary.")
            self.__label = "default"
            self.__position = (0.0, 0.0)
            self.__connected_nodes = []
        self.__successive: dict[str: Line] = {}

    # class methods
    # propagate: update the signal modifying its path attribute and
    #   call the successive element propagate method.
    def propagate(self, signal: Signal_information):
        # update signal path
        signal_path = signal.get_path()
        if len(signal_path) > 1:
            successive_line = self.__label + signal_path[1]
            signal.update_path(self.__label)
            self.__successive[successive_line].propagate(signal)
        else:
            return

    # class getters

    def get_label(self) -> str:
        return self.__label

    def get_position(self) -> tuple[float, float]:
        return self.__position

    def get_connected_nodes(self) -> list[str]:
        return self.__connected_nodes

    def get_successive(self) -> dict:
        return self.__successive

    # class setters

    def set_label(self, new_label):
        try:
            self.__label = str(new_label)
        except ValueError:
            print("The value given to the set_label method was not a string.")

    def set_position(self, new_position):
        try:
            self.__position = tuple(new_position)
        except ValueError:
            print("The value given to the set_position method was not a valid tuple.")

    def set_connected_nodes(self, new_connected_nodes):
        try:
            self.__connected_nodes = list(new_connected_nodes)
        except ValueError:
            print("The value given to the set_connected_nodes method was not a valid list.")

    def set_successive(self, new_successive):
        try:
            self.__successive = dict(new_successive)
        except ValueError:
            print("The value given to the set_successive method was not a valid dictionary.")

    # class overloads

    def __repr__(self):
        return "Node object"

    def __str__(self):
        message = "Node with label: " + self.__label + "\nPosition: (" + \
            "%.3f" % self.__position[0] + ", " + "%.3f" % self.__position[1] + ")\nConnected nodes: " + \
            ", ".join(self.__connected_nodes) + "\n"
        if self.__successive:
            successive_labels = self.__successive.keys()
            message += "Successive lines: " + ", ".join(successive_labels) + "\n"
        return message


# Line class
class Line:
    def __init__(self, label: str = "default", length: float = 0.0):
        self.__label: str = label
        self.__length: float = length
        self.__successive: dict[str: Node] = {}
        self.__state: int = 1   # 1 = free, 0 = occupied

    # class methods
    def latency_generation(self, signal: Signal_information):
        added_latency = self.__length/2e08
        signal.update_latency(added_latency)

    def noise_generation(self, signal: Signal_information):
        generated_noise = 1e-9 * signal.get_signal_power() * self.__length
        signal.update_noise_power(generated_noise)

    # propagate: update the signal information modifying its noise power
    #   and its latency and call the successive element propagate method.
    def propagate(self, signal: Signal_information):
        # update signal information
        self.noise_generation(signal)
        self.latency_generation(signal)
        # call next propagate method
        successive_node = self.__label[1]
        self.__successive[successive_node].propagate(signal)

    # class getters

    def get_label(self) -> str:
        return self.__label

    def get_length(self) -> float:
        return self.__length

    def get_successive(self) -> dict[str: Node]:
        return self.__successive

    def get_state(self) -> int:
        return self.__state

    # class setters

    def set_label(self, new_label):
        try:
            self.__label = str(new_label)
        except ValueError:
            print("The value given to the set_label method was not a string.")

    def set_length(self, new_length):
        try:
            self.__length = float(new_length)
        except ValueError:
            print("The value given to the set_length method was not a float number.")

    def set_successive(self, new_successive):
        try:
            self.__successive = dict(new_successive)
        except ValueError:
            print("The value given to the set_successive method was not a valid dictionary.")

    def set_state(self, new_state):
        try:
            self.__state = int(new_state)
        except ValueError:
            print("The value given to the set_state method was not an int number.")

    # class overloads

    def __repr__(self):
        return "Line object"

    def __str__(self):
        successive_nodes = self.__successive.keys()
        message = "Line with label: " + self.__label + "\nLength: " + \
            "%.3f" % self.__length + " m\nState: " + str(self.__state) + "\n"
        if self.__successive:
            message += "Successive nodes: " + ", ".join(successive_nodes) + "\n"
        return message


class Connection:
    def __init__(self, inout: list[str, str], signal_power=1e-3):
        self.__input:   str = inout[0]
        self.__output:  str = inout[1]
        self.__signal_power: float = signal_power
        self.__latency: float = 0.0
        self.__snr: float = 0.0

    # class methods

    # class getters

    def get_input(self) -> str:
        return self.__input

    def get_output(self) -> str:
        return self.__output

    def get_signal_power(self) -> float:
        return self.__signal_power

    def get_latency(self) -> float:
        return self.__latency

    def get_snr(self) -> float:
        return self.__snr

    # class setters

    def set_input(self, new_input):
        try:
            self.__input = str(new_input)
        except ValueError:
            print("The value given to the set_input method was not a string.")

    def set_output(self, new_output):
        try:
            self.__output = str(new_output)
        except ValueError:
            print("The value given to the set_output method was not a string.")

    def set_signal_power(self, new_signal_power):
        try:
            self.__signal_power = float(new_signal_power)
        except ValueError:
            print("The value given to the set_signal_power method was not a float number.")

    def set_latency(self, new_latency):
        try:
            if new_latency == -1:
                self.__latency = None
            else:
                self.__latency = float(new_latency)
        except ValueError:
            print("The value given to the set_latency method was not a float number.")

    def set_snr(self, new_snr):
        try:
            self.__snr = float(new_snr)
        except ValueError:
            print("The value given to the set_snr method was not a float number.")

    # class overloads

    def __repr__(self):
        return "Connection object"

    def __str__(self):
        message = "Connection between: " + self.__input + " -> " + self.__output + \
                    "\nSignal power: " + "%.3f" % self.__signal_power + " W\n" + \
                    "Latency: " + "%.3f" % self.__latency + " s\nSNR: " + "%.3f" % self.__snr + " dB\n"
        return message


class Network:
    def __init__(self, file_path: str = None):
        if file_path is None:
            self.__nodes: dict[str: Node] = {}
            self.__lines: dict[str: Line] = {}
        else:
            self.__nodes, self.__lines = self.parse_json_to_elements(file_path)
        self.__all_paths: list[list[str]] = []
        self.__weighted_paths: pd.DataFrame = pd.DataFrame()

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

    def propagate(self, signal: Signal_information):
        starting_node = signal.get_path()[0]
        self.__nodes[starting_node].propagate(signal)

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

    def network_analysis(self):     # added "Free" column
        paths_for_pd = []
        for path in self.__all_paths:
            is_free = self.is_path_free(path)
            test_signal = Signal_information(signal_power_value=1e-3)
            test_signal.set_path(path)
            self.propagate(test_signal)     # test signal now has updated latency and noise power
            snr = 10*log10(test_signal.get_signal_power()/test_signal.get_noise_power())
            paths_for_pd.append(
                    [
                        "->".join(path),
                        is_free,
                        test_signal.get_latency(),
                        test_signal.get_noise_power(),
                        snr
                    ]
                )
        self.__weighted_paths = pd.DataFrame(
            paths_for_pd,
            columns=["Path", "Free", "Latency [s]", "Noise Power [W]", "SNR [dB]"]
        )

    def find_best_snr(self, input_node: str, output_node: str) -> str:
        paths_subset = self.__weighted_paths[
            (self.__weighted_paths["Path"].str.startswith(input_node)) &
            (self.__weighted_paths["Path"].str.endswith(output_node)) &
            (self.__weighted_paths["Free"] == 1)
        ]
        if paths_subset.empty:
            return "empty"
        max_index = paths_subset["SNR [dB]"].idxmax(axis=0)
        return str(self.__weighted_paths.iloc[max_index]["Path"])

    def find_best_latency(self, input_node: str, output_node: str) -> str:
        paths_subset = self.__weighted_paths[
            (self.__weighted_paths["Path"].str.startswith(input_node)) &
            (self.__weighted_paths["Path"].str.endswith(output_node)) &
            (self.__weighted_paths["Free"] == 1)
        ]
        if paths_subset.empty:
            return "empty"
        min_index = paths_subset["Latency [s]"].idxmin(axis=0)
        return str(self.__weighted_paths.iloc[min_index]["Path"])

    def set_path_state(self, path: list[str], new_state: int):
        self.__weighted_paths.loc[self.__weighted_paths["Path"] == "->".join(path), "Free"] = int(new_state)

    def set_all_paths_state(self, new_state: int):
        self.__weighted_paths["Free"] = new_state

    def evaluate_path_state(self, path: str):
        is_free = self.is_path_free(path.split("->"))
        self.set_path_state(path.split("->"), is_free)

    def occupy_lines_from_path(self, path: list[str]):
        lines = self.lines_in_path(path)
        for line in lines:
            self.__lines[line].set_state(0)

    def occupy_all_subpaths(self, path: str):
        self.__weighted_paths.loc[self.__weighted_paths["Path"].str.contains(path), "Free"] = 0

    def stream(self, connection_list: list[Connection], best="latency"):
        for connection in connection_list:
            if best == "latency":
                path = self.find_best_latency(connection.get_input(), connection.get_output()).split("->")
            elif best == "snr":
                path = self.find_best_snr(connection.get_input(), connection.get_output()).split("->")
            else:
                return
            if path[0] == "empty":
                connection.set_latency(-1)
                connection.set_snr(0.0)
            else:
                test_signal = Signal_information(signal_power_value=connection.get_signal_power(), given_path=path)
                self.propagate(test_signal)
                self.occupy_all_subpaths("->".join(path))
                self.occupy_lines_from_path(path)
                connection.set_latency(test_signal.get_latency())
                connection.set_snr(test_signal.get_snr())

    # class getters

    def get_nodes(self) -> dict[str: Node]:
        return self.__nodes

    def get_lines(self) -> dict[str: Line]:
        return self.__lines

    def get_all_paths(self) -> list[list[str]]:
        return self.__all_paths

    def get_weighted_paths(self) -> pd.DataFrame:
        return self.__weighted_paths

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
        message = "Network with " + str(len(self.__nodes)) + " nodes and " + str(len(self.__lines)) + " lines\n"
        for node in self.__nodes.values():
            message += str(node)
        for line in self.__lines.values():
            message += str(line)
        message += "\nWeighted paths:\n" + str(self.__weighted_paths)
        return message
