from joblib import load
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


# """Data set reading"""
# df = pd.read_csv("../dataset/processed_kidney_disease.csv")
# X = df.iloc[:, :-1]
# y = df['class']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=45)
#
#
# # train the model with inbuilt classifier
# def train():
#     model = LogisticRegression(n_jobs=-1)
#     model.fit(X_train, y_train)
#
#     predictions = model.predict(X_test)
#     print(accuracy_score(y_test, predictions))
#     dump(model, '../model/lr_model_inbuilt.joblib')
#     print('Model saved')
#
#
# # do prediction from the saved model
# def prediction(data, plot=True):
#     model = load('../model/lr_model_inbuilt.joblib')
#     predictions = model.predict(data)
#
#     if plot:
#         plot_confusion_matrix(model, data, predictions)
#         plt.show()
#     return predictions


def controller_predict(controller, test_data, test_labels, plot=True):
    clf = load('model/lr_model_inbuilt.joblib')
    predictions = clf.predict(test_data)
    if plot:
        plot_confusion_matrix(clf, test_data, test_labels)
        plt.show()

    controller.setLRInbuilt(round(accuracy_score(test_labels, predictions) * 100, 3))

# train()
# y_pred = prediction(X_test)
# print(accuracy_score(y_test, y_pred))
