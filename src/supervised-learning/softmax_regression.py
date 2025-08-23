import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets


class SoftmaxRegression:
    def __init__(
        self, learning_rate=1e-2, penalty="l2", max_iter=1000, mu=0.01, verbose=100
    ):
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.max_iter = max_iter
        self.mu = mu
        self.weights = None
        self.bias = None
        self.verbose = verbose

    def _init_params(self):
        self.weights = np.zeros((self.X_train.shape[1], self.n_classes))
        self.bias = np.zeros(self.n_classes)

    def _one_hot_encode(self, Y):
        one_hot = np.zeros((Y.shape[0], self.n_classes))
        one_hot[np.arange(Y.shape[0]), Y] = 1
        return one_hot

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.n_classes = np.unique(y).shape[0]

        self.y_train = self._one_hot_encode(self.y_train)
        self._init_params()
        for iter in range(self.max_iter + 1):
            # predict
            P = self._softmax(X @ self.weights + self.bias)
            # get gradients
            dw, db = self._get_gradients(P)
            # gradient descent
            self._update_params(dw, db)
            loss = self._cross_entropy(
                self.X_train, self.y_train, self.weights, self.bias
            )
            if iter % self.verbose == 0:
                print(f"Iter {iter}==> Loss = {loss}")

    def _get_gradients(self, preds):
        # Cross entropy gradient
        error = preds - self.y_train
        dw = (1 / self.X_train.shape[0]) * np.dot(self.X_train.T, error)
        if self.penalty == "l2":
            dw += self.mu * self.weights
        db = (1 / self.X_train.shape[0]) * np.sum(error, axis=0)
        return dw, db

    def _update_params(self, dw, db):
        # gradient descent
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
        test = np.asarray(X)
        Z = test @ self.weights + self.bias
        return np.argmax(self._softmax(Z), axis=1)

    def _softmax(self, Z):
        # Z = X @ W (feature * weights) + bias
        # stablized to not overflow after the exponential
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        return np.exp(Z_stable) / np.sum(np.exp(Z_stable), axis=1, keepdims=True)

    def _cross_entropy(self, X, Y, W, bias):
        # log stabilized with epsilon
        epsilon = 1e-9
        Z = X @ W + bias
        n = Y.shape[0]
        loss = -np.sum(Y * np.log(self._softmax(Z) + epsilon)) / n
        return loss


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
    logreg = SoftmaxRegression()
    logreg.fit(X_train_transformed, y_train)
    preds = logreg.predict(X_test_transformed)
    print("actual:")
    print(y_test)
    print("preds:")
    print(preds)
    print(accuracy_score(y_test, preds))

    sklog = LogisticRegression()
    sklog.fit(X_train_transformed, y_train)
    preds2 = sklog.predict(X_test_transformed)
    print("Sklearn:")
    print(accuracy_score(y_test, preds2))
