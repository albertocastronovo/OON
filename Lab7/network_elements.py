from math import log10, floor
from copy import deepcopy
import scipy.constants as spc
from numpy import exp


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
        print("get_snr signal power: " + str(self.__signal_power))
        print("get_snr noise power: " + str(self.__noise_power))
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


class Lightpath(Signal_information):
    def __init__(self, signal_power_value=1e-3, given_path=None, selected_channel=1):
        super(Lightpath, self).__init__(signal_power_value, given_path)
        self.__channel: int = selected_channel
        self.__gsnr: float = 0.0
        self.__rs: float = 32       # signal symbol rate, e+9
        self.__df: float = 50       # frequency spacing between consecutive channels, e+9

    def get_channel(self) -> int:
        return self.__channel

    def get_gsnr(self) -> float:
        return self.__gsnr

    def get_rs(self) -> float:
        return self.__rs

    def get_df(self) -> float:
        return self.__df

    def set_channel(self, new_channel: int):
        try:
            self.__channel = int(new_channel)
        except ValueError:
            print("The value given to the set_channel method was not an int number.")

    def set_gsnr(self, new_gsnr: float):
        try:
            self.__gsnr = int(new_gsnr)
        except ValueError:
            print("The value given to the set_gsnr method was not a float number.")

    def __repr__(self):
        return "Lightpath object"

    def __str__(self):
        message = super(Lightpath, self).__str__()
        message += "Channel: " + str(self.__channel) + "\nGSNR: " + "%.3f" % self.__gsnr + "\n"
        return message


# Node class
#
class Node:
    def __init__(self, node_dictionary=None):
        try:
            self.__label:   str = node_dictionary["label"]
            self.__position: tuple[float, float] = tuple(node_dictionary["position"])
            self.__connected_nodes: list[str] = node_dictionary["connected_nodes"]
            self.__switching_matrix: dict[str: dict[str: list[int]]] = node_dictionary["switching_matrix"]
            self.__static_matrix: dict[str: dict[str: list[int]]] = node_dictionary["switching_matrix"]
            self.__transceiver: str = node_dictionary.get("transceiver", "fixed_rate")
        except KeyError:
            print("Invalid node dictionary.")
            self.__label = "default"
            self.__position = (0.0, 0.0)
            self.__connected_nodes = []
        self.__successive: dict[str: Line] = {}

    # class methods
    # propagate: update the signal modifying its path attribute and
    #   call the successive element propagate method.
    def propagate(self, signal: Lightpath, n_channels):
        # update signal path
        signal_path = signal.get_path()
        if len(signal_path) > 1:
            successive_line = self.__label + signal_path[1]

            optimal_power = self.__successive[successive_line].optimized_launch_power(signal, n_channels)
            signal.set_signal_power(optimal_power)

            signal.update_path(self.__label)
            self.__successive[successive_line].propagate(signal, n_channels)
        else:
            return

    def probe(self, signal: Lightpath, n_channels):
        signal_path = signal.get_path()
        if len(signal_path) > 1:
            successive_line = self.__label + signal_path[1]
            signal.update_path(self.__label)
            self.__successive[successive_line].probe(signal, n_channels)
        else:
            return

    def evaluate_switching_matrix(self, node_1: str, node_2: str, channel: int) -> int:
        return self.__switching_matrix[node_1][node_2][channel-1]

    def occupy_switching_matrix(self, node_1: str, node_2: str, channel: int):
        channels = len(self.__switching_matrix[node_1][node_2])
        self.__switching_matrix[node_1][node_2][channel-1] = 0
        if channel < channels:
            self.__switching_matrix[node_1][node_2][channel] = 0
        if channel > 0:
            self.__switching_matrix[node_1][node_2][channel-2] = 0

    def restore_switching_matrix(self):
        self.__switching_matrix = deepcopy(self.__static_matrix)

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
        if new_transceiver in ["fixed_rate", "flex_rate", "shannon"]:
            self.__transceiver = new_transceiver
        else:
            print("The value given to the set_transceiver method was not a valid transceiver type.")

    # class overloads

    def __repr__(self):
        return "Node object"

    def __str__(self):
        message = "Node with label: " + self.__label + "\nPosition: (" + \
            "%.3f" % self.__position[0] + ", " + "%.3f" % self.__position[1] + ")\nConnected nodes: " + \
            ", ".join(self.__connected_nodes) + "\n"
        if self.__successive:
            successive_labels = self.__successive.keys()
            message += "Successive lines: " + ", ".join(successive_labels) + "\nSwitching matrix: " + \
                str(self.__switching_matrix) + "\n"
        return message


# Line class
class Line:
    def __init__(self, label: str = "default", length: float = 0.0):
        self.__label: str = label
        self.__length: float = length
        self.__successive: dict[str: Node] = {}
        self.__state: list[int] = []   # 1 = free, 0 = occupied
        self.__n_amplifiers: int = floor(length / 80000.0)
        self.__gain_db: float = 16.0
        self.__noise_figure_db: float = 5.5
        self.__gamma: float = 1.27e-3
        self.__alpha_db_km = 0.2
        self.__alpha = self.__alpha_db_km/(20.0*log10(exp(1)))
        self.__effective_length = 1.0/(2.0 * self.__alpha)
        self.__beta: float = 2.13   # e-26

    # class methods
    def latency_generation(self, signal: Lightpath):
        added_latency = self.__length/2e08
        signal.update_latency(added_latency)

    def noise_generation(self, signal: Lightpath, n_channels):
        # generated_noise = 1e-9 * signal.get_signal_power() * self.__length
        generated_noise = self.ase_generation() + self.nli_generation(signal, n_channels)
        print("generated noise: " + str(generated_noise))
        signal.update_noise_power(generated_noise)

    def optimized_launch_power(self, signal: Lightpath, n_channels):
        n_nli = self.evaluate_n_nli(signal, n_channels)
        print("launch power n_nli: " + str(n_nli))
        p_ase = self.ase_generation()
        k = p_ase/(2.0*n_nli)
        if k > 0:
            return k**(1.0/3.0)
        else:
            return (abs(k)**(1.0/3.0))*(-1)

    def evaluate_n_nli(self, signal: Lightpath, n_channels):
        return (16.0 / (27.0 * spc.pi)) * \
            log10( 1000.0 *
            (spc.pi ** 2) * self.__beta * (signal.get_rs() ** 2) * 1e-8 *
            (n_channels ** (2.0 * signal.get_rs() / signal.get_df())) / (2.0 * self.__alpha)
            ) * \
            self.__alpha * (self.__gamma ** 2) * (self.__effective_length ** 2) / \
            (self.__beta * 10.0 * ((signal.get_rs()) ** 3))

    def ase_generation(self):
        f = 193.414 / 10.0                                  # frequency fixed at the C-band center (THz)
        bn = 12.5 / 10.0                                    # noise bandwidth (GHz)
        nf_l = 10**(self.__noise_figure_db/10.0) / 10.0     # noise figure of amplifiers in linear units
        g_l = 10**(self.__gain_db/10.0)                     # gain of amplifiers in linear units
        return self.__n_amplifiers * (
            6.62607015e-10 * f * bn * nf_l * (g_l - 1)      # spc.h is the Planck constant
        )

    def nli_generation(self, signal: Lightpath, n_channels):
        bn = 12.5e+9
        n_nli = self.evaluate_n_nli(signal, n_channels)
        n_span = self.__n_amplifiers + 1
        return ((signal.get_signal_power())**3) * bn * n_nli * n_span

    # propagate: update the signal information modifying its noise power
    #   and its latency and call the successive element propagate method.
    def propagate(self, signal: Lightpath, n_channels):
        # update signal information
        self.noise_generation(signal, n_channels)
        self.latency_generation(signal)
        self.__state[signal.get_channel()-1] = 0    # the channel is now occupied
        # call next propagate method
        successive_node = self.__label[1]
        self.__successive[successive_node].propagate(signal, n_channels)

    def probe(self, signal: Lightpath, n_channels):
        self.noise_generation(signal, n_channels)
        self.latency_generation(signal)
        successive_node = self.__label[1]
        self.__successive[successive_node].probe(signal, n_channels)

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

    def get_n_amplifiers(self) -> int:
        return self.__n_amplifiers

    # class setters

    def set_label(self, new_label):
        try:
            self.__label = str(new_label)
        except ValueError:
            print("The value given to the set_label method was not a string.")

    def set_length(self, new_length):
        try:
            self.__length = float(new_length)
            self.set_n_amplifiers(floor(new_length/80000.0))
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

    def set_n_amplifiers(self, new_n_amplifiers: int):
        try:
            self.__n_amplifiers = new_n_amplifiers
        except ValueError:
            print("The value given to the set_n_amplifiers method was not an int number.")

    # class overloads

    def __repr__(self):
        return "Line object"

    def __str__(self):
        successive_nodes = self.__successive.keys()
        message = "Line with label: " + self.__label + "\nLength: " + \
            "%.3f" % self.__length + " m\nState: " + ", ".join([str(i) for i in self.__state]) + "\n"
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
        message = "Connection between: " + self.__input + " -> " + self.__output + \
                    "\nSignal power: " + "%.3f" % self.__signal_power + " W\n" + \
                    "Latency: " + "%.3f" % self.__latency + " s\nSNR: " + "%.3f" % self.__snr + " dB\n"
        return message
