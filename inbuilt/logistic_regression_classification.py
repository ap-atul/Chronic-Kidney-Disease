import pandas as pd
from joblib import load, dump
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


# train the model with inbuilt classifier
def train():
    """Data set reading"""
    df = pd.read_csv("../dataset/train.csv.csv")
    X = df.iloc[:, :-1]
    y = df['class']

    model = LogisticRegression(n_jobs=-1)
    model.fit(X, y)
    dump(model, '../model/lr_model_inbuilt.joblib')
    print('Model saved')


# do prediction from the saved model
def prediction(data, plot=True):
    model = load('../model/lr_model_inbuilt.joblib')
    predictions = model.predict(data)

    if plot:
        plot_confusion_matrix(model, data, predictions)
        plt.show()
    return predictions


def controller_predict(controller, test_data, test_labels, plot=True):
    clf = load('model/lr_model_inbuilt.joblib')
    predictions = clf.predict(test_data)
    if plot:
        plot_confusion_matrix(clf, test_data, test_labels)
        plt.show()

    controller.setLRInbuilt(round(accuracy_score(test_labels, predictions) * 100, 3))
