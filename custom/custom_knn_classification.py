import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from lib.KNN_Classifier import KNN

# """Data set reading"""
# df = pd.read_csv("../dataset/processed_kidney_disease.csv")
# X = df.iloc[:, :-1].values
# y = df['class'].values
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=45)


# Training the model from the data set
# def train():
#     clf = KNN()
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test, k=9)
#     print("Accuracy with inbuilt :: ", accuracy_score(y_test, y_pred) * 100)
#     dump(clf, '../model/knn_model_custom_train_k_9.joblib')
#     print("Model saved.................")
#
#
# # Predicting using the saved model
# def prediction(test_data):
#     clf = load('../model/knn_model_custom_train_k_9.joblib')
#     predictions = clf.predict(test_data)
#     return predictions


def controller_predict(controller, test_data, test_labels):
    clf = load('model/knn_model_inbuilt_k_9.joblib')
    predictions = clf.predict(test_data)
    controller.setKNNCustom(round(accuracy_score(test_labels, predictions) * 100, 3))


# train()
# y_pred = prediction(X_test)
# print(accuracy_score(y_test, y_pred))
