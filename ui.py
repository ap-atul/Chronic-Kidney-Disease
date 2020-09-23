import os.path
import sys

from PyQt5 import QtGui
from PyQt5.Qt import *
from joblib import dump, load

from controller import Controller

dark = True

"""
Pro tip: Use Ctrl + D to change the theme ;)
"""


class App(QWidget):

    def __init__(self):
        global dark
        super().__init__()

        if os.path.isfile('settings.prop'):
            dark = load('settings.prop')

        self.setGeometry(50, 50, 450, 450)
        self.setMinimumHeight(500)
        self.setMinimumWidth(500)
        self.setWindowTitle('Chronic Kidney Disease')
        self.setWindowIcon(QtGui.QIcon('icons/logo.png'))
        self.show()

        theme = QAction(QIcon('icons/dark_mode.png'), 'Change Theme', self)
        theme.setShortcut('Ctrl+D')
        theme.triggered.connect(self.changeTheme)
        self.addAction(theme)

        self.controller = Controller(self)
        self.greenPalette = QPalette()
        self.greenPalette.setColor(QPalette.Button, QColor(12, 109, 26))

        self.redPalette = QPalette()
        self.redPalette.setColor(QPalette.Button, QColor(139, 0, 0))

        if dark:
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.black)
            self.setPalette(palette)
        self.initUI()

    def initUI(self):
        # ************* ALL THE LAYOUTS ************** #
        # main layout (vertical)
        mainLayout = QVBoxLayout()
        mainLayout.setSpacing(0)

        # CSV layout (vertical)
        csvLayout = QVBoxLayout()
        csvLayout.setSpacing(0)

        # CSV File layout
        csvFileLayout = QHBoxLayout()
        csvFileLayout.setContentsMargins(10, 0, 10, 10)

        # algo layout
        algoLayout = QVBoxLayout()

        # algorithms info layout
        algoInfoLayout = QGridLayout()
        algoInfoLayout.setSpacing(15)
        algoInfoLayout.setContentsMargins(10, 0, 10, 30)

        # bottom close button layout
        closeButtonLayout = QHBoxLayout()
        closeButtonLayout.addStretch(1)
        closeButtonLayout.setSpacing(10)

        # ************* ALL THE WIDGETS ************** #
        # csv label
        csvInfoLabel = QLabel()
        csvInfoLabel.setText("Disease dataset")
        csvInfoLabel.setFont(QtGui.QFont('Ubuntu', 12, QtGui.QFont.Bold))
        csvLayout.addWidget(csvInfoLabel)

        # file icon label
        fileIconLabel = QLabel()
        csvPic = QPixmap('icons/csv.png')
        csvPic = csvPic.scaledToWidth(55)
        fileIconLabel.setContentsMargins(20, 0, 0, 5)
        fileIconLabel.setStyleSheet("background-color : #C98BFD")
        fileIconLabel.setPixmap(csvPic)
        csvFileLayout.addWidget(fileIconLabel)

        # file info label
        fileInfoLabel = QLabel()
        fileInfoLabel.setText("processed_kidney_disease.csv")
        fileInfoLabel.setFont(QtGui.QFont('Arial', 15, QtGui.QFont.Bold))
        fileInfoLabel.setContentsMargins(0, 0, 20, 5)
        fileInfoLabel.setStyleSheet("background-color : #C98BFD; color : #000000")
        csvFileLayout.addWidget(fileInfoLabel)
        csvLayout.addLayout(csvFileLayout)

        visualizeLay = QHBoxLayout()
        # visualizeLay.addStretch(1)
        visualizeLay.setContentsMargins(10, 0, 10, 5)
        visualizeLay.setSpacing(20)
        datasetInfo = QLabel()
        datasetInfo.setText("159 Rows")
        datasetInfo.setStyleSheet("color : #499C54")
        visualizeLay.addWidget(datasetInfo)

        # visualize button
        visualizeButton = QPushButton()
        visualizeButton.setText("Visualize")
        visualizeButton.setMinimumHeight(30)
        visualizeButton.clicked.connect(self.visualize)
        visualizeLay.addWidget(visualizeButton)
        csvLayout.addLayout(visualizeLay)

        # algo label
        algoInfoLabel = QLabel()
        algoInfoLabel.setText("Classifications")
        algoInfoLabel.setFont(QtGui.QFont('Ubuntu', 12, QtGui.QFont.Bold))
        algoInfoLabel.setContentsMargins(0, 5, 0, 0)
        algoLayout.addWidget(algoInfoLabel)

        # KNN algorithm details
        self.knnCustomLabel = QLabel()
        algoInfoLayout.addWidget(self.knnCustomLabel, 0, 0)

        self.knnInbuiltLabel = QLabel()
        algoInfoLayout.addWidget(self.knnInbuiltLabel, 1, 0)

        knnButton = QPushButton()
        knnButton.setText("K Nearest Neighbour")
        knnButton.setMinimumHeight(30)
        knnButton.clicked.connect(self.KNN)
        algoInfoLayout.addWidget(knnButton, 2, 0)

        # NB algorithm details
        self.nbCustomLabel = QLabel()
        algoInfoLayout.addWidget(self.nbCustomLabel, 0, 1)

        self.nbInbuiltLabel = QLabel()
        algoInfoLayout.addWidget(self.nbInbuiltLabel, 1, 1)

        nbButton = QPushButton()
        nbButton.setText("Naive Bayes")
        nbButton.setMinimumHeight(30)
        nbButton.clicked.connect(self.NB)
        algoInfoLayout.addWidget(nbButton, 2, 1)

        # LR algorithm details
        self.lrCustomLabel = QLabel()
        algoInfoLayout.addWidget(self.lrCustomLabel, 0, 2)

        self.lrInbuiltLabel = QLabel()
        algoInfoLayout.addWidget(self.lrInbuiltLabel, 1, 2)

        lrButton = QPushButton()
        lrButton.setText("Logistic Regression")
        lrButton.setMinimumHeight(30)
        lrButton.clicked.connect(self.LR)
        algoInfoLayout.addWidget(lrButton, 2, 2)
        algoLayout.addLayout(algoInfoLayout)

        # comparison details
        self.comparisonDetails = QLabel()
        self.comparisonDetails.setContentsMargins(0, 10, 50, 10)
        self.comparisonDetails.setFont(QtGui.QFont('Ubuntu', 12, QtGui.QFont.Bold))
        self.comparisonDetails.setStyleSheet("color : #DF4A16")
        closeButtonLayout.addWidget(self.comparisonDetails)

        # compare button
        compareButton = QPushButton()
        compareButton.setText("Compare")
        compareButton.clicked.connect(self.compare)
        compareButton.setPalette(self.greenPalette)
        compareButton.update()
        closeButtonLayout.addWidget(compareButton)

        # close button
        closeButton = QPushButton()
        closeButton.setText("Close")
        closeButton.setPalette(self.redPalette)
        closeButton.clicked.connect(self.close)
        closeButtonLayout.addWidget(closeButton)

        # add all components to mail layout
        mainLayout.addLayout(csvLayout)
        mainLayout.addLayout(algoLayout)
        mainLayout.addLayout(closeButtonLayout)

        # add main layout to the window
        self.setLayout(mainLayout)

    def visualize(self):
        self.controller.plot_dataframe()
        return

    def KNN(self):
        self.controller.callKNNs()

    def NB(self):
        self.controller.callNBs()

    def LR(self):
        self.controller.callLRs()

    def setKNNInbuilt(self, text):
        self.knnInbuiltLabel.setText(text)

    def setKNNCustom(self, text):
        self.knnCustomLabel.setText(text)

    def setNBInbuilt(self, text):
        self.nbInbuiltLabel.setText(text)

    def setNBCustom(self, text):
        self.nbCustomLabel.setText(text)

    def setLRInbuilt(self, text):
        self.lrInbuiltLabel.setText(text)

    def setLRCustom(self, text):
        self.lrCustomLabel.setText(text)

    def compare(self):
        details = self.controller.compare()
        self.comparisonDetails.setText(details[0][0] + " > " +
                                       details[1][0] + " > " +
                                       details[2][0])

    def changeTheme(self):
        dump(not dark, "settings.prop")
        self.__init__()


def main():
    app = QApplication(sys.argv)
    application = App()
    application.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
