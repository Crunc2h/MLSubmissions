import numpy as np
WEIGHT_INIT_MODIFIER = 0.1


class Dense:
    def __init__(self, n_features, n_neurons, activation_func, derivative_func, random_bias_init=False, load=False, w=None, b=None):
        if load is False:
            self.W, self.B = (WEIGHT_INIT_MODIFIER * np.random.randn(n_features, n_neurons),
                              np.zeros((1, n_neurons))if random_bias_init is False else np.random.randn(1, n_neurons))
        else:
            self.W, self.B = w, b

        self.a, self.z, self.x = None, None, None
        self.activation_func = activation_func
        self.derivative_func = derivative_func

    def forward(self, x):
        self.x = x
        self.z = self.calculate_output(self.x)
        self.a = self.calculate_activation(self.z)
        return self.a

    def backward(self, delta, alpha):
        d_w = np.dot(self.x.T, delta)
        d_b = np.sum(delta, axis=0)
        self.W -= (alpha * d_w)
        self.B -= (alpha * d_b)

    def calculate_output(self, x):
        return np.dot(x, self.W) + self.B

    def calculate_activation(self, z):
        return self.activation_func(z)
