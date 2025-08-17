import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from scipy.stats import norm

class GaussianNB:
    def __init__(self):
        pass
    def fit(self,X,y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.__calculate_prob()

    def predict(self,X):
        test = np.asarray(X)
        results = np.empty(test.shape[0])
        for i,row in enumerate(test):
            most_likely = 0
            most_prob = -1
            for key,value in self.stats.items():
                curr_prob = value[0]
                for feature in range(1,len(value)-1):
                    class_prob = norm.pdf(row[feature],value[feature][0],value[feature][1])
                    curr_prob *= class_prob
                print("currprob:",curr_prob)
                if curr_prob>most_prob:
                    most_prob=curr_prob
                    most_likely=key
            results[i] = most_likely
        return results


    def __calculate_prob(self):
        unique,count = np.unique(self.y_train, return_counts=True)
        self.stats = {}
        total_count = self.y_train.shape[0]
        for i,class_label in enumerate(unique):
            self.stats[class_label] = []
            # append prior
            self.stats[class_label].append(float(count[i]/total_count))
            # select rows where y_train == class_label
            class_rows = self.X_train[self.y_train == class_label]
            for feature_index in range(self.X_train.shape[1]):
                feature_values = class_rows[:, feature_index]
                mean = np.mean(feature_values)
                std = np.std(feature_values)
                self.stats[class_label].append((mean, std))

        

if __name__=="__main__":
    dataset_dict = {
    'Rainfall': [0.0, 2.0, 7.0, 18.0, 3.0, 3.0, 0.0, 1.0, 0.0, 25.0, 0.0, 18.0, 9.0, 5.0, 0.0, 1.0, 7.0, 0.0, 0.0, 7.0, 5.0, 3.0, 0.0, 2.0, 0.0, 8.0, 4.0, 4.0],
    'Temperature': [29.4, 26.7, 28.3, 21.1, 20.0, 18.3, 17.8, 22.2, 20.6, 23.9, 23.9, 22.2, 27.2, 21.7, 27.2, 23.3, 24.4, 25.6, 27.8, 19.4, 29.4, 22.8, 31.1, 25.0, 26.1, 26.7, 18.9, 28.9],
    'Humidity': [85.0, 90.0, 78.0, 96.0, 80.0, 70.0, 65.0, 95.0, 70.0, 80.0, 70.0, 90.0, 75.0, 80.0, 88.0, 92.0, 85.0, 75.0, 92.0, 90.0, 85.0, 88.0, 65.0, 70.0, 60.0, 95.0, 70.0, 78.0],
    'WindSpeed': [2.1, 21.2, 1.5, 3.3, 2.0, 17.4, 14.9, 6.9, 2.7, 1.6, 30.3, 10.9, 3.0, 7.5, 10.3, 3.0, 3.9, 21.9, 2.6, 17.3, 9.6, 1.9, 16.0, 4.6, 3.2, 8.3, 3.2, 2.2],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
}
    df = pd.DataFrame(dataset_dict)

    # Set feature matrix X and target vector y
    X, y = df.drop(columns='Play'), df['Play'].apply(lambda x: 0 if x=="No" else 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5,random_state=10)      

    nb = GaussianNB()
    nb.fit(X_train,y_train)
    print(nb.stats)
    preds = nb.predict(X_test)
    print("actual:")
    print(y_test)
    print("preds:")
    print(preds)
    print(accuracy_score(y_test,preds))

