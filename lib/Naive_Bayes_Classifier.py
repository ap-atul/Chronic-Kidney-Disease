import numpy as np

"""Bayes’ Theorem provides a way that we can calculate the probability
 of a piece of data belonging to a given class, 
 given our prior knowledge. Bayes’ Theorem is stated as:

    P(class|data) = (P(data|class) * P(class)) / P(data)

Where P(class|data) is the probability of class given the provided data.
"""


class NaiveBayes:
    def __init__(self, X, y):
        """
        initialization, storing the shapes
        :param X: training features
        :param y: training labels
        """
        self.classes_prior = {}
        self.classes_variance = {}
        self.classes_mean = {}
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.eps = 1e-6

    def fit(self, X, y):
        """
        training function that runs and calculates mean
        and variance for each set
        :param X: training features
        :param y: training label
        :return: self.object
        """

        for c in range(self.num_classes):
            X_c = X[y == c]

            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        """
        prediction function that calculates the prob
        from the mean and var calculated
        Note:: using log to minimize the error
        :param X: test set
        :return: list of predictions
        """
        self.num_examples, self.num_features = X.shape
        probs = np.zeros((self.num_examples, self.num_classes))

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self.density_function(
                X, self.classes_mean[str(c)], self.classes_variance[str(c)]
            )
            probs[:, c] = probs_c + np.log(prior)

        return np.argmax(probs, 1)

    def density_function(self, x, mean, sigma):
        """
        using the Guassian Distribution to map the probs
        :param x: row set
        :param mean: the current mean
        :param sigma: variance
        :return: returns the prob
        """
        # Calculate probability from Gaussian density function
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(sigma + self.eps)
        )
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs
