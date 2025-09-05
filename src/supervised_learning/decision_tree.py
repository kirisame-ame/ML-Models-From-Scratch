import numpy as np
from sklearn.tree import DecisionTreeClassifier as skDec
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets


class Question:
    def __init__(self, col, value):
        # col here is the index
        self.col = col
        self.value = value

    def match(self, row):
        row_val = row[self.col]
        return row_val >= self.value


class DecisionNode:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.question = None
        self.true_branch = None
        self.false_branch = None
        self.is_leaf = False

    def build_tree(self):
        gain, question = self._find_best_question()
        if gain == 0:
            self.is_leaf = True
            return
        self.question = question
        true_rows, false_rows = self._partition(self.X, question)
        self.true_branch = DecisionNode(self.X[true_rows], self.y[true_rows])
        self.false_branch = DecisionNode(self.X[false_rows], self.y[false_rows])
        self.true_branch.build_tree()
        self.false_branch.build_tree()

    def _find_best_question(self):
        best_gain = 0
        best_question = None
        curr_impurity = self._gini(self.y)
        for col in range(self.X.shape[1]):
            values = np.unique(self.X[:, col])
            for val in values:
                curr_question = Question(col, val)
                true_rows, false_rows = self._partition(self.X, curr_question)
                if len(true_rows) == 0 or len(false_rows) == 0:
                    # skip question if empty node is produced
                    continue
                curr_gain = self._info_gain(
                    self.y[true_rows], self.y[false_rows], curr_impurity
                )
                if curr_gain >= best_gain:
                    best_gain, best_question = curr_gain, curr_question
        return best_gain, best_question

    def _partition(self, X, question: Question):
        """Returns the index of the rows partioned"""
        true_rows, false_rows = [], []
        for i, row in enumerate(X):
            if question.match(row):
                true_rows.append(i)
            else:
                false_rows.append(i)
        return true_rows, false_rows

    def _gini(self, y):
        # gini impurity
        _, count = np.unique(y, return_counts=True)
        impurity = 1
        total_count = y.shape[0]
        for label_count in count:
            prob = label_count / total_count
            impurity -= prob**2
        return impurity

    def _info_gain(self, left, right, impurity):
        """
        Calculate the information gain (reduction in gini impurity) from a potential split in the dataset.
        Args:
            left (Array-like): The subset of data resulting from the split (left branch).
            right (Array-like): The subset of data resulting from the split (right branch).
            impurity (float): The impurity of the parent node before the split.

        Returns:
            float: The information gain obtained by the split.
        """
        p = float(left.shape[0] / (left.shape[0] + right.shape[0]))
        return impurity - p * self._gini(left) - (1 - p) * self._gini(right)


class DecisionTreeClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.root = DecisionNode(self.X_train, self.y_train)
        self.root.build_tree()

    def predict(self, X):
        test = np.asarray(X)
        results = np.empty(test.shape[0])
        for i, row in enumerate(test):
            probs = self._classify(row, self.root)
            most_likely = np.argmax(probs[:, 1])
            results[i] = probs[most_likely, 0]
        return results

    def _classify(self, row, node: DecisionNode):
        if node.is_leaf:
            values, counts = np.unique(node.y, return_counts=True)
            total_count = node.y.shape[0]
            probs = np.empty((values.shape[0], 2))
            for i, value in enumerate(values):
                probs[i][0] = value
                probs[i][1] = counts[i] / total_count
            return probs
        else:
            if node.question.match(row):
                return self._classify(row, node.true_branch)
            else:
                return self._classify(row, node.false_branch)


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
    dec = DecisionTreeClassifier()
    dec.fit(X_train_transformed, y_train)
    preds = dec.predict(X_test_transformed)
    print("actual:")
    print(y_test)
    print("preds:")
    print(np.unique(preds))
    print(accuracy_score(y_test, preds))

    skdec = skDec()
    skdec.fit(X_train_transformed, y_train)
    preds2 = skdec.predict(X_test_transformed)
    print("Sklearn:")
    print(accuracy_score(y_test, preds2))
