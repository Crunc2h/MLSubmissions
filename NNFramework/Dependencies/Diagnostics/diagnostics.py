import numpy as np


class Diagnostics:
    @staticmethod
    def one_hot_cce_softmax_accuracy(target_labels, network_output):
        count = 0
        row_length = len(target_labels)
        for f in range(row_length):
            if np.argmax(target_labels[f], axis=0) == np.argmax(network_output[f], axis=0):
                count += 1
        return count / len(target_labels)

    @staticmethod
    def one_hot_bce_sigmoid_accuracy(y_labels, network_output):
        count = 0
        for i in range(len(y_labels)):
            if y_labels[i] == 1 and network_output[i, :] >= 0.5 or y_labels[i] == 0 and network_output[i, :] < .5:
                count += 1
        return count
