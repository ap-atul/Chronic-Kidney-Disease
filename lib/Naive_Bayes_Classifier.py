import numpy as np

"""
Bayes’ Theorem provides a way that we can calculate the probability
 of a piece of data belonging to a given class, 
 given our prior knowledge. Bayes’ Theorem is stated as:

    `P(class|data) = (P(data|class) * P(class)) / P(data)`

Where P(class|data) is the probability of class given the provided data.
"""


class NaiveBayes:
    """
    Naive Bayes Classifier

    Attributes
    ----------
    classes_prior : dict
        initial classes (unique)
    classes_variance : dict
        variances of all the input training samples
    classes_mean : dict
        mean of all input training samples
    num_examples : int
        number of examples in training samples
    num_features : int
        number of features in training samples
    num_classes : int
        unique labels
    eps : float
        error
    """

    def __init__(self, X, y):
        self.classes_prior = {}
        self.classes_variance = {}
        self.classes_mean = {}
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.eps = 1e-6

    def fit(self, X, y):
        """
        Training function that runs and calculates mean
        and variance for each set

        Parameters
        ----------
        X : array
            training features
        y : array
            training labels
        """

        for c in range(self.num_classes):
            X_c = X[y == c]

            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        """
        Prediction function that calculates the prob
        from the mean and var calculated

        Parameters
        ----------
        X : array
            testing samples
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
        Using the Guassian Distribution to map the probs

        Parameters
        ----------
        x : array
            testing sample
        mean : float
            mean of the training set
        sigma : float
            variance of the training set

        Returns
        -------
        prob : float
            probability of the current sample
        """
        # Calculate probability from Gaussian density function
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(sigma + self.eps)
        )
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs
