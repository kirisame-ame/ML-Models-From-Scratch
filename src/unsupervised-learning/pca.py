import numpy as np
from sklearn.decomposition import PCA as skPCA
from sklearn.cluster import DBSCAN as skdb
from sklearn.preprocessing import PowerTransformer
from sklearn import datasets


class PCA:
    """SVD based PCA"""

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        U, Vt = self._svd_flip(U, Vt)
        components = Vt
        explained_variance = (S**2) / (X.shape[0])
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        if self.n_components is not None:
            components = components[: self.n_components, :]
            explained_variance = explained_variance[: self.n_components]
            explained_variance_ratio = explained_variance_ratio[: self.n_components]
        self.components = components
        self.explained_variance = explained_variance
        self.explained_variance_ratio = explained_variance_ratio

    def _svd_flip(self, u, v):
        # rows of v, columns of u
        max_abs_v_rows = np.argmax(np.abs(v), axis=1)
        shift = np.arange(v.shape[0])
        indices = max_abs_v_rows + shift * v.shape[1]
        signs = np.sign(np.take(np.reshape(v, (-1,)), indices, axis=0))
        if u is not None:
            u *= signs[np.newaxis, :]
        v *= signs[:, np.newaxis]

        return u, v

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

    def inverse_transform(self, X_transformed):
        return np.dot(X_transformed, self.components.T) + self.mean

    def fit_transform(self, X):
        X = np.asarray(X)
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    iris = datasets.load_iris(as_frame=True)

    # Set feature matrix X and target vector y
    X = iris.data
    pt = PowerTransformer(standardize=True)
    X_transformed = pt.fit_transform(X)

    pca = PCA(n_components=2)
    preds = pca.fit_transform(X_transformed)
    print(preds[:5])
    print(pca.explained_variance_ratio)

    skpca = skPCA(n_components=2)
    skpreds = skpca.fit_transform(X_transformed)
    print(skpreds[:5])
    print(skpca.explained_variance_ratio_)
