import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score

from lib.Logistic_Regression_Classifier import LR


# train the model with inbuilt classifier
def train():
    """Data set reading"""
    df = pd.read_csv("../dataset/train.csv")
    X = df.iloc[:, :-1].values
    y = df['class'].values

    model = LR()
    model.fit(X, y)
    dump(model, '../model/lr_model_custom.joblib')
    print('Model saved')


# do prediction from the saved model
def prediction(data):
    model = load('../model/lr_model_custom.joblib')
    predictions = model.predict(data)
    return predictions


def controller_predict(controller, test_data, test_labels):
    # test_labels = test_labels.values
    clf = load('model/lr_model_custom.joblib')
    predictions = clf.predict(test_data)
    controller.setLRCustom(round(accuracy_score(test_labels, predictions) * 100, 3))
