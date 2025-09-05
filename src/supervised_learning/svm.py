import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets


class BinarySVMClassifier:
    def __init__(
        self,
        learning_rate=0.001,
        lambda_param=0.01,
        n_iters=1000,
        kernel="linear",
        gamma=0.5,
    ):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.gamma = gamma
        self.w = None
        self.b = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y
        if self.kernel == "linear":
            self.w = np.zeros(n_features)
            self.b = 0
            for _ in range(self.n_iters):
                for i, x_i in enumerate(X):
                    self._update_weights(i, x_i, y)
        elif self.kernel == "rbf":
            self.alpha = np.zeros(n_samples)
            self.b = 0
            K = self._compute_kernel_matrix(X)
            for _ in range(self.n_iters):
                for i in range(n_samples):
                    condition = y[i] * (np.sum(self.alpha * y * K[:, i]) + self.b) < 1
                    if condition:
                        self.alpha[i] += self.lr
                        self.b += self.lr * y[i]

    def _rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def _compute_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._rbf_kernel(X[i], X[j])
        return K

    def _update_weights(self, i, x_i, y):
        # margin check valid or not
        condition = y[i] * (np.dot(x_i, self.w) + self.b) >= 1
        if condition:
            self.w -= self.lr * (2 * self.lambda_param * self.w)
        else:
            self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[i]))
            self.b += self.lr * y[i]

    def predict(self, X):
        if self.kernel == "linear":
            approx = np.dot(X, self.w) + self.b
            return np.where(approx >= 0, 1, -1)
        elif self.kernel == "rbf":
            y_pred = []
            for x in X:
                s = 0
                for i in range(len(self.X_train)):
                    s += (
                        self.alpha[i]
                        * self.y_train[i]
                        * self._rbf_kernel(self.X_train[i], x)
                    )
                s += self.b
                y_pred.append(1 if s >= 0 else -1)
            return np.array(y_pred)


class SVMClassifier:
    def __init__(
        self,
        learning_rate=0.001,
        lambda_param=0.01,
        n_iters=1000,
        kernel="linear",
        gamma=0.5,
    ):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.gamma = gamma
        self.classifiers = {}

    def fit(self, X, y):
        self.classifiers = {}
        X = np.asarray(X)
        y = np.asarray(y)
        self.unique = np.unique(y)
        self.class_to_idx = {c: idx for idx, c in enumerate(self.unique)}
        self.n_classes = self.unique.shape[0]
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                mask = (y == self.unique[i]) | (y == self.unique[j])
                X_pair = X[mask]
                y_pair = y[mask]
                y_pair = np.where(y_pair == self.unique[i], 1, -1)
                clf = BinarySVMClassifier(
                    learning_rate=self.lr,
                    n_iters=self.n_iters,
                    lambda_param=self.lambda_param,
                    kernel=self.kernel,
                    gamma=self.gamma,
                )
                clf.fit(X_pair, y_pair)
                self.classifiers[(self.unique[i], self.unique[j])] = clf

    def predict(self, X):
        X = np.asarray(X)
        votes = np.zeros((X.shape[0], self.n_classes), dtype=int)

        for (i, j), clf in self.classifiers.items():
            preds = clf.predict(X)

            # map back to original class labels
            for idx, p in enumerate(preds):
                if p == 1:
                    votes[idx, self.class_to_idx[i]] += 1
                else:
                    votes[idx, self.class_to_idx[j]] += 1

        # final prediction = class with most votes
        final_preds = self.unique[np.argmax(votes, axis=1)]
        return final_preds


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
    print("Linear SVM:")
    svm_linear = SVMClassifier(kernel="linear")
    svm_linear.fit(X_train_transformed, y_train)
    preds_linear = svm_linear.predict(X_test_transformed)
    print("actual:")
    print(y_test)
    print("preds:")
    print(preds_linear)
    print(accuracy_score(y_test, preds_linear))

    print("RBF SVM:")
    svm_rbf = SVMClassifier(kernel="rbf", gamma=0.5)
    svm_rbf.fit(X_train_transformed, y_train)
    preds_rbf = svm_rbf.predict(X_test_transformed)
    print("actual:")
    print(y_test)
    print("preds:")
    print(preds_rbf)
    print(accuracy_score(y_test, preds_rbf))

    sksvm = SGDClassifier(loss="hinge")
    sksvm.fit(X_train_transformed, y_train)
    preds2 = sksvm.predict(X_test_transformed)
    print("Sklearn:")
    print(accuracy_score(y_test, preds2))
