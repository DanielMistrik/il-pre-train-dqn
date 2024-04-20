import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def gen_dataset_action_freq_graph(data_path):
    """
    Generates a frequency graph for the occurences of each action in the provided dataset
    """
    raw_numpy_input = np.load(data_path, allow_pickle=True)
    actions = [item[1] for item in raw_numpy_input]
    labels, values = zip(*Counter(actions).items())

    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()


if __name__ == '__main__':
    gen_dataset_action_freq_graph('../data/MountainCar-v0_10_000.npy')