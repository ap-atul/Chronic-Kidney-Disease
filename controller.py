import numpy as np
from pandas import read_csv

from custom import custom_logistic_regression_classification, custom_naive_bayes_classification, \
    custom_knn_classification
from inbuilt import naivebayes_classification, logistic_regression_classification, knn_classification
from visualize_data import Visualize

"""
Reads the input csv file and makes a pd.DataFrame object to be used to call all the prediction functions 
inside the Custom and SKLearn models.

After prediction, accuracies are returned and comparision is done.

"""

df = read_csv("./dataset/test.csv")
X = df.iloc[:, :-1]
y = df['class']

knn = []
nb = []
lr = []


class Controller:
    def __init__(self, App):
        """
        Controller to control the UI interactions and communication to set data and call functions on button
        clicks

        Parameters
        ----------
        App : QWidget
            to call UI functions to set text on labels
        """
        self.app = App

    def plot_dataframe(self):
        """
        Calling the Visualize.visualize() function to plot the data frame
        """
        print("Plotting")
        visualizer = Visualize(df)
        visualizer.visualize()
        return

    def callKNNs(self):
        """
        Calling both implementations of SKLearn and Custom KNN
        """
        knn_classification.controller_predict(self, X, y)
        custom_knn_classification.controller_predict(self, X.values, y.values)

    def callNBs(self):
        """
        Calling both implementations of SKLearn and Custom NB
        """
        naivebayes_classification.controller_predict(self, X, y)
        custom_naive_bayes_classification.controller_predict(self, X, y)

    def callLRs(self):
        """
        Calling both implementations of SKLearn and Custom LR
        """
        logistic_regression_classification.controller_predict(self, X, y)
        custom_logistic_regression_classification.controller_predict(self, X.values, y.values)

    def setKNNInbuilt(self, text):
        """
        Setting the accuracies to the UI using the App object
        """
        knn.append(text)
        self.app.knnInbuiltLabel.setText("Sklearn : " + str(text))

    def setKNNCustom(self, text):
        """
        Setting the accuracies to the UI using the App object
        """
        knn.append(text)
        self.app.knnCustomLabel.setText("Custom : " + str(text))

    def setNBInbuilt(self, text):
        """
        Setting the accuracies to the UI using the App object
        """
        nb.append(text)
        self.app.nbInbuiltLabel.setText("Sklearn : " + str(text))

    def setNBCustom(self, text):
        """
        Setting the accuracies to the UI using the App object
        """
        nb.append(text)
        self.app.nbCustomLabel.setText("Custom : " + str(text))

    def setLRInbuilt(self, text):
        """
        Setting the accuracies to the UI using the App object
        """
        lr.append(text)
        self.app.lrInbuiltLabel.setText("Sklearn : " + str(text))

    def setLRCustom(self, text):
        """
        Setting the accuracies to the UI using the App object
        """
        lr.append(text)
        self.app.lrCustomLabel.setText("Custom : " + str(text))

    def compare(self):
        """
        Setting the comparison of all the algorithms to the UI using the App object
        """
        k = np.mean(knn)
        n = np.mean(nb)
        l = np.mean(lr)
        dict = {"KNN": k, "NB": n, "LR": l}
        return sorted(dict.items(), key=lambda x: x[1], reverse=True)
