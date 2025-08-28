import numpy as np
from collections import deque
from sklearn.cluster import DBSCAN as skdb
from sklearn.preprocessing import PowerTransformer
from sklearn import datasets


class DBSCAN:
    def __init__(self, epsilon=0.5, min_samples=5, metric="euclidean", p=2):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.metric = metric
        self.p = p

    def fit(self, X):
        self.X = np.asarray(X)
        n_samples = self.X.shape[0]
        self.labels = -1 * np.ones(n_samples)
        visited = np.zeros(n_samples, dtype=bool)
        id_cluster = 0
        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._get_neighbors(X, i, self.p)
            if neighbors.shape[0] >= self.min_samples:
                self._create_cluster(self.X, i, neighbors, id_cluster, visited)
                id_cluster += 1
        return self

    def _get_neighbors(self, X, i, p):
        if self.metric == "euclidean":
            distances = np.linalg.norm(X - X[i], axis=1)
        elif self.metric == "minkowski":
            distances = np.sum(np.abs(X - X[i]) ** p, axis=1) ** (1 / p)
        elif self.metric == "manhattan":
            distances = np.sum(np.abs(X - X[i]), axis=1)
        return np.where(distances <= self.epsilon)[0]

    def _create_cluster(self, X, i, neighbors, id_cluster, visited):
        queue = deque(neighbors)
        self.labels[i] = id_cluster

        while queue:
            neighbor_idx = queue.popleft()
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._get_neighbors(X, neighbor_idx, self.p)
                if new_neighbors.shape[0] >= self.min_samples:
                    queue.extend(new_neighbors)
            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = id_cluster

    def fit_predict(self, X):
        self.fit(X)
        return self.labels


if __name__ == "__main__":
    iris = datasets.load_iris(as_frame=True)

    # Set feature matrix X and target vector y
    X = iris.data
    pt = PowerTransformer(standardize=True)
    X_transformed = pt.fit_transform(X)

    dbscan = DBSCAN()
    preds = dbscan.fit_predict(X_transformed)
    print(preds)

    skdb = skdb()
    skpreds = skdb.fit_predict(X_transformed)
    print(skpreds)
