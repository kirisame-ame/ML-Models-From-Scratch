import numpy as np
import math
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None

    def dim(self):
        return self.data.shape


class loss:
    def get_loss(self, input, target):
        raise NotImplementedError

    def get_gradient(self, input, target):
        raise NotImplementedError


class cross_entropy(loss):
    """
    Accepts input in the form of probablities
    """

    def get_loss(self, input, target):
        N = input.shape[0]
        correct_logprobs = -np.log(input[range(N), target] + 1e-9)
        return np.mean(correct_logprobs)

    def get_gradient(self, input, target):
        N = input.shape[0]
        grad = input.copy()
        grad[range(N), target] -= 1
        grad /= N
        return grad


class MSE(loss):
    """
    Works for both classification (with integer labels)
    and regression (with continuous targets).
    """

    def get_loss(self, input, target):
        # Classification case → one-hot encode
        if target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1):
            one_hot = np.zeros((target.shape[0], input.shape[1]))
            one_hot[np.arange(target.shape[0]), target.flatten()] = 1
            target = one_hot

        # Regression case → directly compare
        return np.mean((input - target) ** 2)

    def get_gradient(self, input, target):
        # Classification case → one-hot encode
        if target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1):
            one_hot = np.zeros((target.shape[0], input.shape[1]))
            one_hot[np.arange(target.shape[0]), target.flatten()] = 1
            target = one_hot

        # Gradient is the same formula in both cases
        return 2 * (input - target) / input.shape[0]


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
    def forward(self, x: Tensor):
        raise NotImplementedError

    def backward(self, grad, lr):
        raise NotImplementedError


class Relu(Layer):
    def forward(self, x):
        self.mask = (x.data > 0).astype(float)
        return Tensor(x.data * self.mask, requires_grad=x.requires_grad)

    def backward(self, grad, lr):
        return grad * self.mask


class Sigmoid(Layer):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x.data))
        return Tensor(self.out, requires_grad=x.requires_grad)

    def backward(self, grad, lr):
        return grad * self.out * (1 - self.out)


class Softmax(Layer):
    def forward(self, x):
        Z_stable = x.data - np.max(x.data, axis=1, keepdims=True)
        z_softmax = np.exp(Z_stable) / np.sum(np.exp(Z_stable), axis=1, keepdims=True)
        self.out = z_softmax
        return Tensor(z_softmax, requires_grad=x.requires_grad)

    def backward(self, grad, lr):
        dot = np.sum(grad * self.out, axis=1, keepdims=True)
        grad_input = self.out * (grad - dot)
        return grad_input


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(np.empty((out_features, in_features)))
        self.bias = Tensor(np.empty(out_features)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        init.kaiming_uniform(self.weight)
        if self.bias is not None:
            init.kaiming_uniform(tensor=self.bias, fan_mode="fan_out")

    def forward(self, X: Tensor) -> Tensor:
        self.X = X
        y = X.data @ self.weight.data.T
        if self.bias is not None:
            y = y + self.bias.data
        return Tensor(y, requires_grad=X.requires_grad or self.weight.requires_grad)

    def backward(self, grad, lr):
        dw = grad.T @ self.X.data
        db = np.sum(grad, axis=0)
        dx = grad @ self.weight.data
        self.weight.data -= lr * dw
        if self.bias is not None:
            self.bias.data -= lr * db
        return dx


class Conv2D(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride_length=(1, 1),
        use_padding=False,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride_length = stride_length  # (stride_y, stride_x)
        self.use_padding = use_padding
        # weight shape: (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = (
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        )
        self.bias = np.zeros(out_channels)

    def forward(self, x: Tensor):
        # x shape: (batch, in_channels, height, width)
        batch_size, in_channels, height, width = x.data.shape
        k = self.kernel_size
        sy, sx = self.stride_length
        if self.use_padding:
            pad_y = ((height - 1) * sy + k - height) // 2
            pad_x = ((width - 1) * sx + k - width) // 2
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (pad_y, pad_y), (pad_x, pad_x)),
                mode="constant",
            )
            height_p, width_p = x_padded.shape[2], x_padded.shape[3]
        else:
            x_padded = x.data
            height_p, width_p = height, width
        out_height = (height_p - k) // sy + 1
        out_width = (width_p - k) // sx + 1
        out = np.zeros((batch_size, self.out_channels, out_height, out_width))
        for b in range(batch_size):
            for oc in range(self.out_channels):
                conv_sum = np.zeros((height_p, width_p))
                for ic in range(in_channels):
                    img = x_padded[b, ic]
                    kernel = self.weight[oc, ic]
                    kernel_padded = np.zeros_like(img)
                    kernel_padded[:k, :k] = kernel
                    img_fft = np.fft.fft2(img)
                    kernel_fft = np.fft.fft2(kernel_padded)
                    conv_fft = img_fft * kernel_fft
                    conv = np.fft.ifft2(conv_fft).real
                    conv_sum += conv
                # Strided crop to valid region
                for i in range(out_height):
                    for j in range(out_width):
                        out[b, oc, i, j] = (
                            conv_sum[i * sy : i * sy + k, j * sx : j * sx + k].sum()
                            / (k * k)
                            + self.bias[oc]
                        )
        self.x = x_padded
        return Tensor(out, requires_grad=x.requires_grad)

    def backward(self, grad, lr):
        # grad shape: (batch, out_channels, out_height, out_width)
        batch_size, in_channels, height, width = self.x.shape
        k = self.kernel_size
        out_height = height - k + 1
        out_width = width - k + 1
        # Gradients
        grad_weight = np.zeros_like(self.weight)
        grad_bias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(self.x)
        for b in range(batch_size):
            for oc in range(self.out_channels):
                grad_bias[oc] += np.sum(grad[b, oc])
                for ic in range(in_channels):
                    for i in range(out_height):
                        for j in range(out_width):
                            region = self.x[b, ic, i : i + k, j : j + k]
                            grad_weight[oc, ic] += grad[b, oc, i, j] * region
                            grad_input[b, ic, i : i + k, j : j + k] += (
                                grad[b, oc, i, j] * self.weight[oc, ic]
                            )
        # Update weights and bias
        self.weight -= lr * grad_weight
        self.bias -= lr * grad_bias
        return grad_input


class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride_length=(2, 2)):
        self.kernel_size = kernel_size
        self.stride_length = stride_length

    def forward(self, x: Tensor):
        # x shape: (batch, channels, height, width)
        batch_size, channels, height, width = x.data.shape
        k = self.kernel_size
        sy, sx = self.stride_length
        out_height = (height - k) // sy + 1
        out_width = (width - k) // sx + 1
        # Use stride tricks for efficient windowing
        shape = (batch_size, channels, out_height, out_width, k, k)
        strides = (
            x.data.strides[0],
            x.data.strides[1],
            x.data.strides[2] * sy,
            x.data.strides[3] * sx,
            x.data.strides[2],
            x.data.strides[3],
        )
        windows = np.lib.stride_tricks.as_strided(
            x.data, shape=shape, strides=strides, writeable=False
        )
        out = np.max(windows, axis=(4, 5))
        self.x = x.data
        return Tensor(out, requires_grad=x.requires_grad)

    def backward(self, grad, lr):
        # grad shape: (batch, channels, out_height, out_width)
        batch_size, channels, height, width = self.x.shape
        k = self.kernel_size
        s = self.stride
        out_height = (height - k) // s + 1
        out_width = (width - k) // s + 1
        grad_input = np.zeros_like(self.x)
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        window = self.x[b, c, i * s : i * s + k, j * s : j * s + k]
                        max_val = np.max(window)
                        mask = window == max_val
                        grad_input[b, c, i * s : i * s + k, j * s : j * s + k] += (
                            grad[b, c, i, j] * mask
                        )
        return grad_input


class Model:
    """
    A simple feedforward neural network model for supervised learning.
    Args:
        layers (list[Layer]): List of Layer objects representing the network architecture.
        loss (loss): Loss function object with methods for computing loss and gradients.
    Methods:
        forward(X: Tensor) -> Tensor:
            Performs a forward pass through the network with input tensor X.
        backward(grad, lr):
            Performs a backward pass (backpropagation) through the network using the provided gradient and learning rate.
        fit(X: Tensor, y: Tensor, epochs=1000,batch_size=32, lr=0.01, verbose=100):
            Trains the model using batch gradient descent for a specified number of epochs.
            Prints loss every 'verbose' epochs.
        predict(X: Tensor):
            Returns the predicted class labels for input tensor X using argmax on the output.
    """

    def __init__(self, layers: list[Layer], loss: loss):
        self.layers = layers
        self.loss = loss

    def forward(self, X: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def fit(
        self, X: Tensor, y: Tensor, epochs=1000, batch_size=32, lr=0.01, verbose=100
    ):
        n_samples = X.data.shape[0]
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                X_batch = Tensor(X.data[batch_idx])
                y_batch = Tensor(y.data[batch_idx])
                # forward prop
                y_pred = self.forward(X_batch)

                # backprop
                grad = self.loss.get_gradient(y_pred.data, y_batch.data)
                self.backward(grad, lr)

            if (epoch + 1) % verbose == 0:
                print(
                    f"Epoch {epoch+1}, Loss: {self.loss.get_loss(y_pred.data,y_batch.data):.6f}"
                )

    def predict(self, X: Tensor):
        out = self.forward(X)
        return np.argmax(out.data, axis=1)


if __name__ == "__main__":
    iris = datasets.load_iris(as_frame=True)

    # Set feature matrix X and target vector y
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=5
    )
    pt = PowerTransformer(standardize=True)  # Standard Scaling already included
    X_train_transformed = pt.fit_transform(X_train)
    X_test_transformed = pt.transform(X_test)
    model = Model(
        layers=[Linear(4, 4), Relu(), Linear(4, 3), Softmax()], loss=cross_entropy()
    )
    model.fit(Tensor(X_train_transformed), Tensor(y_train), epochs=1000, lr=1)
    preds = model.predict(Tensor(X_test_transformed))

    print(accuracy_score(y_test, preds))
    print("actual:")
    print(np.ravel(y_test))
    print("preds:")
    print(preds)

    print("\nTesting Conv2D and MaxPool2D layers:")
    dummy_input = np.random.rand(2, 1, 6, 6)
    tensor_input = Tensor(dummy_input)
    conv = Conv2D(
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        stride_length=(2, 2),
        use_padding=True,
    )
    pool = MaxPool2D(kernel_size=2, stride_length=(2, 2))
    conv_out = conv.forward(tensor_input)
    print("Conv2D output shape:", conv_out.data.shape)
    pool_out = pool.forward(conv_out)
    print("MaxPool2D output shape:", pool_out.data.shape)
