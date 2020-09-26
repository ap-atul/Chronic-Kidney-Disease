import numpy as np
from tqdm import tqdm

"""
Custom implementation of the Logistic Regression classifier
using coefficient calculations and sigmoid function
"""


def addBias(X):
    """
    Adding column of bias to the input data

    Parameters
    ---------
    X : numpy array
        input array

    Returns
    -------
    numpy array
        column gets added
    """
    return np.insert(X, 0, 1, axis=1)


def softMax(z):
    """
    Taking soft max of the vector
    """
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)


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

    def __init__(self, l_rate=0.001, n_epoch=100):
        self.learning_rate = l_rate
        self.n_epoch = n_epoch
        self.weights = None
        self.bias = None
        self.data = None
        self.classes = None
        self.classLabels = None

    def fit(self, X_train, y_train):
        """
        run training and calculate coefficients for
        each such data

        Parameters
        ----------
        X_train : numpy array
            training array
        y_train : numpy array
            training labels
        """
        self.data = addBias(X_train)
        self.classes = np.unique(y_train)
        self.classLabels = {c: i for i, c in enumerate(self.classes)}
        labels = self.hotEncode(y_train)

        self.weights = np.zeros(shape=(len(self.classes), self.data.shape[1]))
        for _ in tqdm(range(self.n_epoch)):
            # y = m*x + c
            y = np.dot(self.data, self.weights.T).reshape(-1, len(self.classes))

            # apply soft max
            y_predicted = softMax(y)

            # compute gradients
            dw = np.dot((y_predicted - labels).T, self.data)

            # update parameters
            self.weights -= self.learning_rate * dw

    def hotEncode(self, y):
        """
        Creating an identity matrix for the labels

        Parameters
        ---------
        y : numpy array
            labels
        """
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.classLabels[c])(y).reshape(-1)]

    def predict(self, X_test):
        """
        Prediction done on the input testing samples

        Parameters
        ---------
        X_test : numpy array
            testing samples
        """
        X_test = addBias(X_test)
        prediction = np.dot(X_test, self.weights.T).reshape(-1, len(self.classes))
        probability = softMax(prediction)
        predictionClass = np.vectorize(lambda c: self.classes[c])(np.argmax(probability, axis=1))
        return predictionClass
