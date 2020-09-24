from collections import Counter

from scipy.spatial import distance
from tqdm import tqdm

"""
K Nearest Neighbour classification
"""


class KNN:
    """
    Classifier for nearest neighbours

    Attributes
    ----------
    X_train : numpy array
        training features to initialize
    Y_train : numpy array
        training lables to initialize
    """

    def fit(self, X_train, Y_train):
        """
        KNN does not have any training phase so just initialization

        Parameters
        ----------
        X_train : numpy array
            training features array
        Y_train : numpy array
            training labels array
        """
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test, k=5):
        """
        Prediction based on training sample evaluation

        Parameters
        ----------
        X_test : numpy array
            testing samples to predict
        k : int
            number of neighbours closest to the point

        Returns
        -------
        predictions : list
            list of prediction labels
        """
        predictions = []
        for row in tqdm(X_test):
            label = self.closest(row, k)
            predictions.append(label)
        return predictions

    def closest(self, row, k):
        """
        Evaluation based on the training samples and its relation with the testing samples

        Parameters
        ----------
        row : array
            single row from the samples
        k : int
            neighbour count

        Returns
        -------
        prediction : int
            prediction calculated
        """
        distances = []
        for i in range(len(self.X_train)):
            distances.append((i, distance.euclidean(row, self.X_train[i])))
        distances = sorted(distances, key=lambda x: x[1])[0:k]
        k_indices = []
        for i in range(k):
            k_indices.append(distances[i][0])
        k_labels = []
        for i in range(k):
            k_labels.append(self.Y_train[k_indices[i]])
        c = Counter(k_labels)
        return c.most_common()[0][0]
