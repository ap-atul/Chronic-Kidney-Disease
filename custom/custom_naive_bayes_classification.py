import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from lib.Naive_Bayes_Classifier import NaiveBayes as NB

# # """Data set reading"""
# df = pd.read_csv("../dataset/processed_kidney_disease.csv")
# X = df.iloc[:, :-1]
# y = df['class']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44)


# train the model with inbuilt classifier
# def train():
#     model = NB(X_train, y_train)
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#     print(accuracy_score(y_test, y_pred))
#     dump(model, '../model/nb_model_custom.joblib')
#     print('Model saved..........')
#
#
# # do prediction from the saved model
def prediction(data):
    model = load('../model/nb_model_custom.joblib')
    predictions = model.predict(data)
    return predictions


def controller_predict(controller, test_data, test_labels):
    clf = load('model/nb_model_custom.joblib')
    predictions = clf.predict(test_data)

    controller.setNBCustom(round(accuracy_score(test_labels, predictions) * 100, 3))

# train()
# y_pred = prediction(X_test)
# print(accuracy_score(y_test, y_pred))
# classification_report(y_test, predictions)
# print(confusion_matrix(y_test, predictions))
