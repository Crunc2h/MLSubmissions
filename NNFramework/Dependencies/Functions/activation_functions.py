import numpy as np


class ActivationFuncs:
    @staticmethod
    def relu(x):
        a = np.maximum(0, x)
        return a

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def softmax(x):
        exp = np.exp(x)
        sum_exp = np.sum(exp, axis=1, keepdims=True)
        return exp / sum_exp

    @staticmethod
    def softmax_derivative(a):
        return a * (1 - a)

    @staticmethod
    def sigmoid_activation(x):
        res = np.divide(1, np.add(1, np.exp(-x)))
        return res

    @staticmethod
    def sigmoid_derivative(x):
        return ActivationFuncs.sigmoid_activation(x) * (1 - ActivationFuncs.sigmoid_activation(x))
