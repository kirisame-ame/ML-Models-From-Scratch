import numpy as np
from scipy.stats import mode
import sklearn.neighbors
class KNeighborsClassifier():
    def __init__(self, k=3, distance_metric='euclidean', p=3):
        self.k = k
        self.distance_metric = distance_metric
        self.p = p

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        results = np.empty(X.shape[0])
        for i,row in enumerate(X):
            distances = np.empty(self.X_train.shape[0])
            for j,row_train in enumerate(self.X_train):
                if self.distance_metric=="euclidean":
                    distances[j] = self.euclidean_distance(row,row_train)
                elif self.distance_metric=="manhattan":
                    distances[j] =self.manhattan_distance(row,row_train)
                elif self.distance_metric=="minkowski":
                    distances[j]=self.minkowski_distance(row,row_train,self.p)
            neighbors = np.argsort(distances)[:self.k]
            neighbor_labels = self.y_train[neighbors]
            #grab the mode, scipy.stats.mode returns mode,count
            result = np.ravel(mode(neighbor_labels))[0]
            results[i] = result
        return results.astype(int)

    def euclidean_distance(self,x1, x2):
        return np.linalg.norm(x1 - x2)
    def manhattan_distance(self,x1, x2):
        return np.sum(np.abs(x1 - x2))
    def minkowski_distance(self,x1, x2, p=3):
        return np.sum(np.abs(x1 - x2) ** p) ** (1/p)
    
if __name__=="__main__":
    print("KNN testing")
    dataset = [[1,1,0],
               [1,3,0],
               [2,4,0],
               [10,10,1],
               [20,20,1],
               [15,15,1]]
    dataset = np.array(dataset)
    X_train = dataset[:,:2]
    y_train =dataset[:,2]
    knn = KNeighborsClassifier()
    knn.fit(X_train,y_train)
    test = np.array([[100,10]])
    print("Scratch:",knn.predict(test))
    skknn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
    skknn.fit(X_train,y_train)
    print("Sklearn:",skknn.predict(test))