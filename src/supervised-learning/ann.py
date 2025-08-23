import numpy as np
import math


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None

    def dim(self):
        return self.data.shape


class init:
    def zeros(tensor: Tensor):
        tensor.data[:] = np.zeros(tensor.dim())

    def ones(tensor: Tensor):
        tensor.data[:] = np.ones(tensor.dim())

    def kaiming_uniform(
        tensor: Tensor, a=math.sqrt(5), nonlinearity="leaky_relu", fan_mode="fan_in"
    ):
        dim = tensor.dim()
        fan = init._get_fan(tensor, fan_mode)
        gain = init._calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        tensor.data[:] = np.random.uniform(-bound, bound, dim)

    def xavier_uniform(tensor: Tensor, gain=1.0):
        dim = tensor.dim()
        fan_in, fan_out = dim[1], dim[0]
        bound = gain * math.sqrt(6 / (fan_in + fan_out))
        tensor.data[:] = np.random.uniform(-bound, bound, dim)

    def _calculate_gain(nonlinearity: str, a):
        nonlinearity = nonlinearity.lower()
        gain = 1
        if nonlinearity == "leaky_relu":
            # results in lecun initialization (1/in_features)
            gain = math.sqrt(2.0 / (1 + a**2))
        elif nonlinearity == "relu":
            gain = math.sqrt(2.0)
        return gain

    def _get_fan(tensor: Tensor, fan_mode):
        dim = tensor.dim()
        if fan_mode == "fan_in":
            fan = dim[1]
        elif fan_mode == "fan_out":
            fan = dim[0]
        return fan


class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad, lr):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(np.empty((in_features, out_features)))
        self.bias = Tensor(np.empty(out_features)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        init.kaiming_uniform(self.weight)
        if self.bias is not None:
            init.kaiming_uniform(tensor=self.bias, fan_mode="fan_out")

    def forward(self, init):
        pass

    def backward(self, grad, lr):
        pass


class Model:
    def __init__(self):
        pass

    def forward(self, X):
        pass

    def backward(self, grad, lr):
        # batch gradient descent
        pass

    def fit(self, X, y, epochs=1000, lr=0.01):
        pass

    def predict(self, X):
        out = self.forward(X)
        return np.argmax(out, axis=1)
