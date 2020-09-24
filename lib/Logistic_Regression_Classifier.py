from math import exp

from tqdm import tqdm

"""
Custom implementation of the Logistic Regression classifier
using coefficient calculations and sigmoid function
"""


class LR:
    """
    Logistic Regression Classifier

    Attributes
    ----------
    l_rate : float
        initial learning rate
    n_epoch : int
        number of iterations
    """

    def __init__(self, l_rate=0.1, n_epoch=100):
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coefficients = []

    def fit(self, X_train, Y_train):
        """
        run training and calculate coefficients for
        each such data

        Parameters
        ----------
        X_train : array
            training array
        Y_train : array
            training labels
        """
        self.coefficients = [0.0 for i in range(len(X_train[0]) + 1)]
        for epoch in tqdm(range(self.n_epoch)):
            for row_ind in range(len(X_train)):
                y_pred = self.predict_cal(X_train[row_ind])
                error = Y_train[row_ind] - y_pred
                self.coefficients[0] = self.coefficients[0] + self.l_rate * error * y_pred * (1.0 - y_pred)
                for i in range(len(X_train[row_ind])):
                    self.coefficients[i + 1] = self.coefficients[i + 1] + \
                                               self.l_rate * error * y_pred * (1.0 - y_pred) * \
                                               X_train[row_ind][i]

    def predict_cal(self, row):
        """
        Run the prediction for each data

        Parameters
        ----------
        row : array
            current row in the testing samples
        """
        u = self.coefficients[0]
        for i in range(len(row) - 1):
            u += self.coefficients[i + 1] * row[i]
        return 1.0 / (1.0 + exp(-u))  # sigmoid function

    def predict(self, X_validation):
        """
        Prediction done on the input testing samples

        Parameters
        ---------
        X_validation : array
            testing samples
        """
        predictions = list()
        # make predictions (validate)
        for row in X_validation:
            y_pred = round(self.predict_cal(row))
            predictions.append(y_pred)
        return predictions
