import matplotlib.pyplot as plt

plt.style.use('ggplot')

"""
Class to visualize the data set of the csv file
Usage create an object pass the data set object
"""


class Visualize:
    def __init__(self, dataFrame):
        """
        pass the pandas data frame read from csv file
        :param dataFrame: dataset
        """
        self.dataFrame = dataFrame

    def visualize(self):
        """
        plot the data set initialized
        :return: None
        """
        self.dataFrame.hist()
        plt.show()
