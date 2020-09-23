import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split

from custom import custom_logistic_regression_classification, custom_naive_bayes_classification, \
    custom_knn_classification
from inbuilt import naivebayes_classification, logistic_regression_classification, knn_classification
from visualize_data import Visualize

df = read_csv("./dataset/processed_kidney_disease.csv")
X = df.iloc[:, :-1]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44)

print(type(X_train))
knn = []
nb = []
lr = []


class Controller:
    def __init__(self, App):
        self.app = App

    def plot_dataframe(self):
        print("Plotting")
        visualizer = Visualize(df)
        visualizer.visualize()
        return

    def callKNNs(self):
        knn_classification.controller_predict(self, X_test, y_test)
        custom_knn_classification.controller_predict(self, X_test, y_test)

    def callNBs(self):
        naivebayes_classification.controller_predict(self, X_test, y_test)
        custom_naive_bayes_classification.controller_predict(self, X_test, y_test)

    def callLRs(self):
        logistic_regression_classification.controller_predict(self, X_test, y_test)
        custom_logistic_regression_classification.controller_predict(self, X_test.values, y_test.values)

    def setKNNInbuilt(self, text):
        knn.append(text)
        self.app.knnInbuiltLabel.setText("Sklearn : " + str(text))

    def setKNNCustom(self, text):
        knn.append(text)
        self.app.knnCustomLabel.setText("Custom : " + str(text))

    def setNBInbuilt(self, text):
        nb.append(text)
        self.app.nbInbuiltLabel.setText("Sklearn : " + str(text))

    def setNBCustom(self, text):
        nb.append(text)
        self.app.nbCustomLabel.setText("Custom : " + str(text))

    def setLRInbuilt(self, text):
        lr.append(text)
        self.app.lrInbuiltLabel.setText("Sklearn : " + str(text))

    def setLRCustom(self, text):
        lr.append(text)
        self.app.lrCustomLabel.setText("Custom : " + str(text))

    def compare(self):
        k = np.mean(knn)
        n = np.mean(nb)
        l = np.mean(lr)
        dict = {"KNN": k, "NB": n, "LR": l}
        return sorted(dict.items(), key=lambda x: x[1], reverse=True)
