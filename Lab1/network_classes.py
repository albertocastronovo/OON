from json import load
from math import sqrt, log10
import matplotlib.pyplot as plt
import pandas as pd
from textwrap import dedent


#   1.  Signal_information class

class Signal_information:
    def __init__(self, signal_power_value=1e-3, given_path=None):
        self.__signal_power: float = signal_power_value
        self.__noise_power: float = 0.0
        self.__latency: float = 0.0
        self.__path: list[str] = []
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
        message = dedent(f"""\
            Signal Information
            Signal power:   {self.__signal_power :.2f} W
            Noise power:    {self.__noise_power :.2f} W
            Latency:        {self.__latency :.2f} s
            Path:           {', '.join(self.__path)}"""
                         )
        return message


#   2.  Node class

class Node:
    def __init__(self, node_dictionary=None):
        try:
            self.__label: str = node_dictionary["label"]
            self.__position: tuple[float, float] = tuple(node_dictionary["position"])
            self.__connected_nodes: list[str] = node_dictionary["connected_nodes"]
        except (KeyError, TypeError):
            print("Invalid node dictionary.")
            self.__label = "default"
            self.__position = (0.0, 0.0)
            self.__connected_nodes = []
        self.__successive: dict[str: Line] = {}

    # class methods

    def propagate(self, signal: Signal_information):
        signal_path = signal.get_path()
        if len(signal_path) > 1:
            successive_line = self.__label + signal_path[1]
            signal.update_path(self.__label)
            self.__successive[successive_line].propagate(signal)
        else:
            return signal

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
        message = dedent(f"""\
                    Node with label:    {self.__label}
                    Position:           {" m, ".join(map(str, self.__position))} m
                    Connected nodes:    {", ".join(self.__connected_nodes)}
                    Successive:         {", ".join(self.__successive.keys())}"""
                         )
        return message


#   3.  Line class

class Line:
    def __init__(self, label: str = "default", length: float = 0.0):
        self.__label: str = label
        self.__length: float = length
        self.__successive: dict[str: Node] = {}

    # class methods

    def latency_generation(self, signal: Signal_information):
        added_latency = self.__length / 2e08
        signal.update_latency(added_latency)

    def noise_generation(self, signal: Signal_information):
        generated_noise = 1e-9 * signal.get_signal_power() * self.__length
        signal.update_noise_power(generated_noise)

    def propagate(self, signal: Signal_information):
        self.noise_generation(signal)
        self.latency_generation(signal)
        successive_node = self.__label[1]
        self.__successive[successive_node].propagate(signal)

    # class getters

    def get_label(self) -> str:
        return self.__label

    def get_length(self) -> float:
        return self.__length

    def get_successive(self) -> dict[str: Node]:
        return self.__successive

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

    # class overloads

    def __repr__(self):
        return "Line object"

    def __str__(self):
        message = dedent(f"""\
                            Line with label:    {self.__label}
                            Length:             {self.__length :.2f} m
                            Successive:         {", ".join(self.__successive.keys())}"""
                         )
        return message


#   4.  Network class

class Network:
    def __init__(self, file_path: str = None):
        if file_path is None:
            self.__nodes: dict[str: Node] = {}
            self.__lines: dict[str: Line] = {}
        else:
            self.__nodes, self.__lines = self.parse_json_to_elements(file_path)
        self.__all_paths: list[list[str]] = []
        self.__dataframe: pd.DataFrame = pd.DataFrame()

    # class methods

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
                        (json_data[key]["position"][0] - json_data[node]["position"][0]) ** 2 +
                        (json_data[key]["position"][1] - json_data[node]["position"][1]) ** 2
                    )
                    output_lines[line_label] = Line(line_label, line_length)

        return output_nodes, output_lines

    def connect(self):
        for node in self.__nodes.values():
            node.set_successive(
                {
                    node.get_label() + connected_node: self.__lines[node.get_label() + connected_node]
                    for connected_node in node.get_connected_nodes()
                }
            )
        for line in self.__lines.values():
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

    def propagate(self, signal: Signal_information) -> Signal_information:
        starting_node = signal.get_path()[0]
        return self.__nodes[starting_node].propagate(signal)

    def draw(self):
        for node in self.__nodes.keys():
            node_pos = self.__nodes[node].get_position()
            plt.scatter(*node_pos, s=500, marker='o', color='r', linewidths=0)
            plt.annotate(node, xy=node_pos, xytext=(-3, -6), textcoords="offset pixels")

        for line in self.__lines.keys():
            node_pos = [self.__nodes[line[0]].get_position(), self.__nodes[line[1]].get_position()]
            plt.plot(*zip(*node_pos), color='b', zorder=-1)

        plt.axis("off")
        plt.title("Network")
        plt.show()

    #   5. Pandas Dataframe

    def network_analysis(self):
        paths_for_pd = []
        for path in self.__all_paths:
            test_signal = Signal_information(signal_power_value=1e-3)
            test_signal.set_path(path)
            self.propagate(test_signal)
            snr = 10 * log10(test_signal.get_signal_power() / test_signal.get_noise_power())
            paths_for_pd.append(
                [
                    "->".join(path),
                    test_signal.get_latency(),
                    test_signal.get_noise_power(),
                    snr
                ]
            )
        self.__dataframe = pd.DataFrame(paths_for_pd, columns=["Path", "Latency [s]", "Noise Power [W]", "SNR [dB]"])

    # class getters

    def get_nodes(self) -> dict[str: Node]:
        return self.__nodes

    def get_lines(self) -> dict[str: Line]:
        return self.__lines

    def get_all_paths(self) -> list[list[str]]:
        return self.__all_paths

    def get_dataframe(self) -> pd.DataFrame:
        return self.__dataframe

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

    def set_dataframe(self, new_dataframe):
        try:
            self.__dataframe = pd.DataFrame(new_dataframe)
        except ValueError:
            print("The value given to the set_dataframe method was not a valid dataframe.")

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
