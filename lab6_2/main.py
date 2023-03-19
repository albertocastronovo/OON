from network import Network
from network_elements import Line, Lightpath
from textwrap import dedent
from math import log10
import matplotlib.pyplot as plt
from numpy import logspace, linspace


def main():
    n = Network("network/not_full_network_flex.json", 6)
    n.connect()
    n.network_analysis()
    # print(n)

    hist_old = []
    hist_new = []
    lengths = logspace(start=3, stop=6, num=1000, base=10)

    for length in lengths:
        ex_line = Line("AB", length)
        ex_line.set_channels(6)
        test_signal = Lightpath(1e-3, ["A", "B"], 1)
        ex_line.noise_generation(test_signal)
        old_noise = test_signal.get_noise_power()
        ase_noise = ex_line.ase_generation()
        nli_noise = ex_line.nli_generation(test_signal)
        hist_old.append(10*log10(1e-3 / old_noise))
        hist_new.append(10*log10(1e-3 / (ase_noise + nli_noise)))
    plt.title("SNR vs line length, 6 channels, 1 mW, 1 km to 1000 km")
    plt.plot(lengths, hist_new, alpha=0.5, label="ASE + NLI")
    plt.plot(lengths, hist_old, alpha=0.5, label="old formula")
    plt.legend(loc="upper right")
    plt.xlabel("Line length [m]")
    plt.ylabel("SNR [dB]")
    plt.show()

    hist2_new = []
    hist2_old = []
    in_power = logspace(start=-6, stop=0, num=100)
    for pow in in_power:
        ex_line = Line("AB", 100000)
        ex_line.set_channels(6)
        test_signal = Lightpath(pow, ["A", "B"], 1)
        ex_line.noise_generation(test_signal)
        old_noise = test_signal.get_noise_power()
        ase_noise = ex_line.ase_generation()
        nli_noise = ex_line.nli_generation(test_signal)
        hist2_old.append(10 * log10(pow / old_noise))
        hist2_new.append(10 * log10(pow / (ase_noise + nli_noise)))

    plt.title("SNR vs input power, 6 channels, 100 km, 1 uW to 1 W")
    plt.plot(in_power, hist2_new, alpha=0.5, label="ASE + NLI")
    plt.plot(in_power, hist2_old, alpha=0.5, label="old formula")
    plt.legend(loc="lower left")
    plt.xlabel("Signal power [W]")
    plt.ylabel("SNR [dB]")
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    main()
