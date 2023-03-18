from math import log10
from textwrap import dedent
from copy import deepcopy


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
        message = dedent(f"""\
                                Signal Information
                                Signal power:   {self.__signal_power :.3f} W
                                Noise power:    {self.__noise_power :.3f} W
                                Latency:        {self.__latency :.3f} s
                                Path:           {', '.join(self.__path)}"""
                         )
        return message


class Lightpath(Signal_information):
    def __init__(self, signal_power_value=1e-3, given_path=None, selected_channel=1):
        super(Lightpath, self).__init__(signal_power_value, given_path)
        self.__channel: int = selected_channel
        self.__last_crossed_node: str = "default"

    def get_channel(self) -> int:
        return self.__channel

    def get_last_crossed_node(self):
        return self.__last_crossed_node

    def set_channel(self, new_channel):
        try:
            self.__channel = int(new_channel)
        except ValueError:
            print("The value given to the set_channel method was not an int number.")

    def set_last_crossed_node(self, new_last_node: str):
        self.__last_crossed_node = new_last_node

    def __repr__(self):
        return "Lightpath object"

    def __str__(self):
        message = super(Lightpath, self).__str__()
        message += f"Channel: {self.__channel}"
        return message


# Node class
#
class Node:
    def __init__(self, node_dictionary=None):
        try:
            self.__label:   str = node_dictionary["label"]
            self.__position: tuple[float, float] = tuple(node_dictionary["position"])
            self.__connected_nodes: list[str] = node_dictionary["connected_nodes"]
            self.__switching_matrix: dict[str: dict[str: list[int]]] = node_dictionary.get("switching_matrix", None)    # 1. switching matrix
            self.__original_matrix: dict[str: dict[str: list[int]]] = node_dictionary.get("switching_matrix", None)
            self.__transceiver: str = node_dictionary.get("transceiver", "fixed-rate")

        except KeyError:
            print("Invalid node dictionary.")
            self.__label = "default"
            self.__position = (0.0, 0.0)
            self.__connected_nodes = []
        self.__successive: dict[str: Line] = {}


    # class methods
    # propagate: update the signal modifying its path attribute and
    #   call the successive element propagate method.
    def propagate(self, signal: Lightpath):
        # update signal path
        signal_path = signal.get_path()
        if len(signal_path) > 1:
            next_node = signal_path[1]
            successive_line = self.__label + next_node
            signal.update_path(self.__label)
            channels = len(list(list(self.__switching_matrix.values())[0].values())[0])
            current_channel = signal.get_channel()
            last_node = signal.get_last_crossed_node()
            if last_node != "default":
                self.set_busy_sm(last_node, next_node, current_channel)
                if current_channel > 1:
                    self.set_busy_sm(last_node, next_node, current_channel - 1)
                if current_channel < channels:
                    self.set_busy_sm(last_node, next_node, current_channel + 1)

            signal.set_last_crossed_node(self.__label)
            self.__successive[successive_line].propagate(signal)

        else:
            signal.set_last_crossed_node(self.__label)
            return

    def probe(self, signal: Lightpath):
        signal_path = signal.get_path()
        if len(signal_path) > 1:
            successive_line = self.__label + signal_path[1]
            signal.update_path(self.__label)
            self.__successive[successive_line].probe(signal)
        else:
            return

    def evaluate_switching_matrix(self, node_1: str, node_2: str, channel: int) -> int:
        return self.__switching_matrix[node_1][node_2][channel-1]

    def set_busy_sm(self, node_1: str, node_2: str, channel: int):
        self.__switching_matrix[node_1][node_2][channel-1] = 0

    def reset_switching_matrix(self):
        self.__switching_matrix = deepcopy(self.__original_matrix)

    # class getters

    def get_label(self) -> str:
        return self.__label

    def get_position(self) -> tuple[float, float]:
        return self.__position

    def get_connected_nodes(self) -> list[str]:
        return self.__connected_nodes

    def get_successive(self) -> dict:
        return self.__successive

    def get_switching_matrix(self) -> dict:
        return self.__switching_matrix

    def get_transceiver(self) -> str:
        return self.__transceiver

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

    def set_switching_matrix(self, new_switching_matrix):
        try:
            self.__switching_matrix = dict(new_switching_matrix)
        except ValueError:
            print("The value given to the set_switching_matrix method was not a valid dictionary.")

    def set_transceiver(self, new_transceiver):
        try:
            self.__transceiver = str(new_transceiver)
        except ValueError:
            print("The value given to the set_transceiver method was not a valid string.")

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


# Line class
class Line:
    def __init__(self, label: str = "default", length: float = 0.0):
        self.__label: str = label
        self.__length: float = length
        self.__successive: dict[str: Node] = {}
        self.__state: list[int] = []   # 1 = free, 0 = occupied

    # class methods
    def latency_generation(self, signal: Signal_information):
        added_latency = self.__length/2e08
        signal.update_latency(added_latency)

    def noise_generation(self, signal: Signal_information):
        generated_noise = 1e-9 * signal.get_signal_power() * self.__length
        signal.update_noise_power(generated_noise)

    # propagate: update the signal information modifying its noise power
    #   and its latency and call the successive element propagate method.
    def propagate(self, signal: Lightpath):
        # update signal information
        self.noise_generation(signal)
        self.latency_generation(signal)
        self.__state[signal.get_channel()-1] = 0    # the channel is now occupied
        # call next propagate method
        successive_node = self.__label[1]
        self.__successive[successive_node].propagate(signal)

    def probe(self, signal: Lightpath):
        self.noise_generation(signal)
        self.latency_generation(signal)
        successive_node = self.__label[1]
        self.__successive[successive_node].probe(signal)

    # class getters

    def get_label(self) -> str:
        return self.__label

    def get_length(self) -> float:
        return self.__length

    def get_successive(self) -> dict[str: Node]:
        return self.__successive

    def get_state(self) -> list[int]:
        return self.__state

    def get_channel_state(self, channel: int) -> int:
        return self.__state[channel-1]

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
            self.__state = list(new_state)
        except ValueError:
            print("The value given to the set_state method was not a valid list.")

    # class overloads

    def __repr__(self):
        return "Line object"

    def __str__(self):
        message = dedent(f"""\
                                Line with label:    {self.__label}
                                Length:             {self.__length :.3f} m
                                Successive:         {", ".join(self.__successive.keys())}"""
                         )
        return message


class Connection:
    def __init__(self, inout: list[str, str], signal_power=1e-3):
        self.__input:   str = inout[0]
        self.__output:  str = inout[1]
        self.__signal_power: float = signal_power
        self.__latency: float = 0.0
        self.__snr: float = 0.0
        self.__bit_rate: float = 0.0

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

    def get_bit_rate(self) -> float:
        return self.__bit_rate

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

    def set_bit_rate(self, new_bit_rate):
        try:
            self.__bit_rate = float(new_bit_rate)
        except ValueError:
            print("The value given to the set_bit_rate method was not a float number.")

    # class overloads

    def __repr__(self):
        return "Connection object"

    def __str__(self):
        if self.__latency is None:
            lat = "None"
        else:
            lat = ".3f".format(self.__latency)
        message = dedent(f"""\
                                Connection between: {self.__input} -> {self.__output}
                                Signal power:       {self.__signal_power :.3f} W
                                Latency:            {lat} s
                                SNR:                {self.__snr :.3f} dB"""
                         )
        return message
