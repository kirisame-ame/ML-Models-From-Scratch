import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets


class VotingClassifier:
    def __init__(self, estimators, voting="soft"):
        self.estimators = estimators
        self.voting = voting

    def fit(self, X, y):
        for est in self.estimators:
            est.fit(X, y)

    def predict(self, X):
        if self.voting == "soft":
            probas = [est.predict_proba(X) for est in self.estimators]
            avg_proba = np.mean(probas, axis=0)
            return np.argmax(avg_proba, axis=1)
        else:
            preds = np.array([est.predict(X) for est in self.estimators])
            # majority vote
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=preds
            )


if __name__ == "__main__":
    from gaussian_naive_bayes import GaussianNB
    from decision_tree import DecisionTreeClassifier
    from softmax_regression import SoftmaxRegression

    iris = datasets.load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=5
    )
    pt = PowerTransformer(standardize=True)
    X_train_transformed = pt.fit_transform(X_train)
    X_test_transformed = pt.transform(X_test)

    clf1 = GaussianNB()
    clf2 = DecisionTreeClassifier()
    clf3 = SoftmaxRegression()
    voting_clf = VotingClassifier([clf1, clf2, clf3], voting="soft")
    voting_clf.fit(X_train_transformed, y_train)
    preds = voting_clf.predict(X_test_transformed)
    print("actual:")
    print(np.ravel(y_test))
    print("preds:")
    print(preds)
    print("Accuracy:", accuracy_score(y_test, preds))
