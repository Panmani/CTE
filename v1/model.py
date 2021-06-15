import pickle
import numpy as np
from sent2vec import EMBED_CACHE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def get_positive_percentage(y):
    return np.sum(y) / len(y)

if __name__ == '__main__':
    with open(EMBED_CACHE, "rb") as embed_file:
        sent_embeddings, labels = pickle.load(embed_file)


    X_train, X_test, y_train, y_test = train_test_split(sent_embeddings, labels, test_size=0.2, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    print("Percent of positive examples in [train] set:", get_positive_percentage(y_train))
    print("Percent of positive examples in [test ] set:", get_positive_percentage(y_test))

    """ MLP """
    MLP_clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    print("MLP [train] acc: ", MLP_clf.score(X_train, y_train))
    print("MLP [test ] acc: ", MLP_clf.score(X_test, y_test))

    """ SVM """
    SVM_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    SVM_clf.fit(X_train, y_train)
    print("SVM [train] acc: ", SVM_clf.score(X_train, y_train))
    print("SVM [test ] acc: ", SVM_clf.score(X_test, y_test))
