import matplotlib.pyplot as plt

plt.style.use('ggplot')

"""
Class to visualize the data set of the csv file
Usage create an object pass the data set object
"""


class Visualize:
    def __init__(self, dataFrame):
        """
        Pass the pandas data frame read from csv file

        Parameters
        ----------
        dataFrame : pd.dataFrame
            pandas data frame object from reading the csv file
        """
        self.dataFrame = dataFrame

    def visualize(self):
        """
        plot the data set initialized
        """
        self.dataFrame.hist()
        plt.show()
