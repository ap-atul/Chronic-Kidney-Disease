import pandas as pd
from joblib import dump
from joblib import load
from sklearn.metrics import accuracy_score

from lib.KNN_Classifier import KNN


# Training the model from the data set
def train():
    """Data set reading"""
    df = pd.read_csv("../dataset/train.csv")
    X = df.iloc[:, :-1].values
    y = df['class'].values

    clf = KNN()
    clf.fit(X, y)
    dump(clf, '../model/knn_model_custom_train_k_9.joblib')
    print("Model saved.................")


# Predicting using the saved model
def prediction(test_data):
    clf = load('../model/knn_model_custom_train_k_9.joblib')
    predictions = clf.predict(test_data)
    return predictions


def controller_predict(controller, test_data, test_labels):
    clf = load('model/knn_model_custom_train_k_9.joblib')
    predictions = clf.predict(test_data)
    controller.setKNNCustom(round(accuracy_score(test_labels, predictions) * 100, 3))
