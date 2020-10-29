import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score

from lib.Naive_Bayes_Classifier import NaiveBayes as NB


# train the model with inbuilt classifier
def train():
    """Data set reading"""
    df = pd.read_csv("../dataset/train.csv")
    X = df.iloc[:, :-1]
    y = df['class']

    model = NB(X, y)
    model.fit(X, y)
    dump(model, '../model/nb_model_custom.joblib')
    print('Model saved..........')


# # do prediction from the saved model
def prediction(data):
    model = load('../model/nb_model_custom.joblib')
    predictions = model.predict(data)
    return predictions


def controller_predict(controller, test_data, test_labels):
    clf = load('model/nb_model_custom.joblib')
    predictions = clf.predict(test_data)

    controller.setNBCustom(round(accuracy_score(test_labels, predictions) * 100, 3))
