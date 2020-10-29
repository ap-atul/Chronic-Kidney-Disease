import pandas as pd
from joblib import load, dump
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

from sklearn.neighbors import KNeighborsClassifier


# Training model using sk learn classifier
def train():
    """Data set reading"""
    df = pd.read_csv("../dataset/train.csv.csv")
    X = df.iloc[:, :-1]
    y = df['class']

    clf = KNeighborsClassifier(n_neighbors=9, n_jobs=-1)
    clf.fit(X, y)
    dump(clf, '../model/knn_model_inbuilt_k_9.joblib')
    print("Model saved.................")


# Predicting using the saved model
def predict_load_model(X_test, plot=True):
    clf = load('../model/knn_model_inbuilt_k_9.joblib')
    predictions = clf.predict(X_test)

    if plot:
        plot_confusion_matrix(clf, X_test, predictions)
        plt.show()
    return predictions


def controller_predict(controller, test_data, test_labels, plot=True):
    clf = load('model/knn_model_inbuilt_k_9.joblib')
    predictions = clf.predict(test_data)

    if plot:
        plot_confusion_matrix(clf, test_data, test_labels)
        plt.show()

    controller.setKNNInbuilt(round(accuracy_score(test_labels, predictions) * 100, 3))
