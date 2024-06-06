import numpy as np


class CostFuncs:
    @staticmethod
    def categorical_cross_entropy(onehot_labels, network_output):
        clipped_output = np.clip(network_output, 1e-7, 1 - 1e-7)
        return np.mean(np.dot(-np.log(clipped_output), onehot_labels.T))

    # BCE = -(y * log(p) + (1 - y) * log(1 - p))
    @staticmethod
    def binary_cross_entropy(onehot_labels, network_output):
        clipped_output = np.clip(network_output, 1e-7, 1 - 1e-7)
        return -(np.mean(np.dot(onehot_labels, np.log(network_output))))

    @staticmethod
    def mean_squared_error(y_labels, network_output):
        return np.mean((network_output - y_labels)**2)