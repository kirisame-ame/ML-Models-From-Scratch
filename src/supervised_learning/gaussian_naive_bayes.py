import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as skNB
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score
from scipy.stats import norm
from sklearn import datasets


class GaussianNB:
    def predict_proba(self, X):
        test = np.asarray(X)
        n_classes = len(self.stats)
        proba = np.zeros((test.shape[0], n_classes))
        class_labels = sorted(self.stats.keys())
        for i, row in enumerate(test):
            log_probs = []
            for key, value in self.stats.items():
                curr_prob = value[0]
                for feature in range(1, len(value) - 1):
                    class_prob = norm.pdf(
                        row[feature], value[feature][0], value[feature][1]
                    )
                    curr_prob += np.log(class_prob)
                log_probs.append(curr_prob)
            # Convert log-probs to probabilities
            probs = np.exp(log_probs - np.max(log_probs))
            probs /= np.sum(probs)
            proba[i] = probs
        return proba

    def __init__(self, smoothing=1e-9):
        self.smoothing = smoothing

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.__calculate_prob()

    def predict(self, X):
        test = np.asarray(X)
        results = np.empty(test.shape[0])
        for i, row in enumerate(test):
            most_likely = 0
            most_prob = -np.inf
            for key, value in self.stats.items():
                curr_prob = value[0]
                for feature in range(1, len(value) - 1):
                    class_prob = norm.pdf(
                        row[feature], value[feature][0], value[feature][1]
                    )
                    curr_prob += np.log(class_prob)
                if curr_prob > most_prob:
                    most_prob = curr_prob
                    most_likely = key
            results[i] = most_likely
        return results.astype(int)

    def __calculate_prob(self):
        unique, count = np.unique(self.y_train, return_counts=True)
        self.stats = {}
        total_count = self.y_train.shape[0]
        max_var = 0
        for i, class_label in enumerate(unique):
            self.stats[class_label] = []
            # append prior
            self.stats[class_label].append(np.log(float(count[i] / total_count)))
            # select rows where y_train == class_label
            class_rows = self.X_train[self.y_train == class_label]
            for feature_index in range(self.X_train.shape[1]):
                feature_values = class_rows[:, feature_index]
                mean = np.mean(feature_values)
                var = np.var(feature_values)
                if var > max_var:
                    max_var = var
                self.stats[class_label].append([mean, var])

        for key, value in self.stats.items():
            for feature in range(1, len(value) - 1):
                value[feature][1] = np.sqrt(
                    self.smoothing * max_var + value[feature][1]
                )


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
    nb = GaussianNB()
    nb.fit(X_train_transformed, y_train)
    print(nb.stats)
    preds = nb.predict(X_test_transformed)
    print("actual:")
    print(y_test)
    print("preds:")
    print(preds)
    print(accuracy_score(y_test, preds))

    sknb = skNB()
    sknb.fit(X_train_transformed, y_train)
    preds2 = sknb.predict(X_test_transformed)
    print("Sklearn:")
    print(accuracy_score(y_test, preds2))
