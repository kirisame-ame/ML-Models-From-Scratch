import numpy as np
from sklearn.cluster import KMeans as skMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer
from sklearn import datasets
import os

os.environ["OMP_NUM_THREADS"] = "1"


class KMeans:
    def __init__(self, n_clusters=5, init="k-means++", max_iter=300):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter

    def fit(self, X):
        self.X = np.asarray(X)
        centroid_idx = self._init_centroids()
        self.centroids = self.X[centroid_idx]

        for i in range(self.max_iter):
            distances = self._calc_distances(self.X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            old_centroids = self.centroids
            self.centroids = np.array(
                [np.mean(X[self.labels == i], axis=0) for i in range(self.n_clusters)]
            )
            if np.allclose(self.centroids, old_centroids):
                break

    def _init_centroids(self):
        if self.init == "random":
            centroid_idx = np.random.choice(
                self.X.shape[0], self.n_clusters, replace=False
            )
            return centroid_idx
        elif self.init == "k-means++":
            centroid_idx = []
            centroid_idx.append(np.random.randint(self.X.shape[0]))
            for i in range(1, self.n_clusters):
                # compute squared distances to nearest centroid
                distances = np.min(
                    self._calc_distances(self.X, self.X[centroid_idx]) ** 2, axis=1
                )
                # choose next centroid with prob proportional to distance^2
                probs = distances / np.sum(distances)
                idx = np.random.choice(self.X.shape[0], p=probs)
                centroid_idx.append(idx)

            return np.array(centroid_idx)

    def _calc_distances(self, X, centroids):
        return np.linalg.norm(X[:, None] - centroids, axis=2)

    def predict(self, X):
        X = np.asarray(X)
        distances = self._calc_distances(X, self.centroids)
        return np.argmin(distances, axis=1)


if __name__ == "__main__":
    iris = datasets.load_iris(as_frame=True)

    # Set feature matrix X and target vector y
    X = iris.data
    pt = PowerTransformer(standardize=True)
    X_transformed = pt.fit_transform(X)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_transformed)
    preds = kmeans.predict(X_transformed)
    print(preds)

    skmeans = skMeans(n_clusters=3)
    skmeans.fit(X_transformed)
    skpreds = skmeans.predict(X_transformed)
    print(skpreds)
