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


# Training model using sk learn classifier
# def train_model():
#     clf = KNeighborsClassifier(n_neighbors=9, n_jobs=-1)
#     clf.fit(X_train, y_train)
#
#     y_pred = clf.predict(X_test)
#     print("Accuracy with inbuilt :: ", accuracy_score(y_test, y_pred) * 100)
#     dump(clf, '../model/knn_model_inbuilt_k_9.joblib')
#     print("Model saved.................")
#
#
# # Predicting using the saved model
# def predict_load_model(X_test, plot=True):
#     clf = load('../model/knn_model_inbuilt_k_9.joblib')
#     predictions = clf.predict(X_test)
#
#     if plot:
#         plot_confusion_matrix(clf, X_test, predictions)
#         plt.show()
#     return predictions


def controller_predict(controller, test_data, test_labels, plot=True):
    clf = load('model/knn_model_inbuilt_k_9.joblib')
    predictions = clf.predict(test_data)
    
    if plot:
        plot_confusion_matrix(clf, test_data, test_labels)
        plt.show()

    controller.setKNNInbuilt(round(accuracy_score(test_labels, predictions) * 100, 3))

# train_model()
# ypred = predict_load_model(X_test, plot=True)
# print(accuracy_score(y_test, ypred))
