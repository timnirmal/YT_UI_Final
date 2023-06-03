import os

import pandas as pd
from joblib import dump, load
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import *

from plot_graphs import GraphPlotter

# show all columns
pd.set_option('display.max_columns', None)
# np.random.seed(42)

model_path = "models/classes/"

tags = ['0', '1', '2', '3', '4', '5']


################################################## Multinomial Naive Bayes model

# NB
def nb_train(X_train, X_test, y_train, y_test):
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB())])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return nb, accuracy, report


# KNN
def knn_train(X_train, X_test, y_train, y_test):
    knn = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', KNeighborsClassifier(n_neighbors=17, p=6, metric='euclidean'))])
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return knn, accuracy, report


# RF
def RF_train(X_train, X_test, y_train, y_test):
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', ensemble.RandomForestClassifier())])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return nb, accuracy, report


# LR
def LR_train(X_train, X_test, y_train, y_test):
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', linear_model.LogisticRegression())])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return nb, accuracy, report


# Svm
def svc_train(X_train, X_test, y_train, y_test):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', svm.SVC()),
                    ])
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return sgd, accuracy, report


# SGD
def svc_train(X_train, X_test, y_train, y_test):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return sgd, accuracy, report


def sgd_train(X_train, X_test, y_train, y_test):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return sgd, accuracy, report


def train(save=True):
    global best_model_name
    path = "classes_dataset.csv"

    # read csv
    df = pd.read_csv(path, encoding='utf8')

    # keep only filtered_sentence, class
    df = df[['filtered_sentence', 'class']]

    print(df.head())

    # print unique values in Class
    print(df['class'].unique())

    # convert Class to numeric   Sports, Religious, Political, Sexual, Education, and Entertainment.
    class_to_num = {
        "class": {
            "Sports": 0,
            "Religious": 1,
            "Political": 2,
            "Sexual": 3,
            "Education": 4,
            "Entertainment": 5
        }
    }

    df = df.replace(class_to_num)

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head())

    # split into train/test sets
    sentences = df['filtered_sentence'].values.astype('U')
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.1, random_state=42)

    # print df types
    print(df.dtypes)

    # Model training
    nb = nb_train(X_train, X_test, y_train, y_test)
    knn = knn_train(X_train, X_test, y_train, y_test)
    RF = RF_train(X_train, X_test, y_train, y_test)
    LR = LR_train(X_train, X_test, y_train, y_test)
    svc = svc_train(X_train, X_test, y_train, y_test)
    sgd = sgd_train(X_train, X_test, y_train, y_test)

    print("Naive Bayes Accuracy: ", nb[1])
    print("KNN Accuracy: ", knn[1])
    print("Random Forest Accuracy: ", RF[1])
    print("Logistic Regression Accuracy: ", LR[1])
    print("SVC Accuracy: ", svc[1])
    print("SGD Accuracy: ", sgd[1])

    best_model = None
    # pick best model and accuracy
    best_model_accuracy = max(nb[1], knn[1], RF[1], LR[1], svc[1], sgd[1])
    print("\nBest Model Accuracy: ", best_model_accuracy)

    # best model name
    if best_model_accuracy == nb[1]:
        print("Best Model: Naive Bayes")
        best_model = nb[0]
        best_model_name = "Naive Bayes"
    elif best_model_accuracy == knn[1]:
        print("Best Model: KNN")
        best_model = knn[0]
        best_model_name = "KNN"
    elif best_model_accuracy == RF[1]:
        print("Best Model: Random Forest")
        best_model = RF[0]
        best_model_name = "Random Forest"
    elif best_model_accuracy == LR[1]:
        print("Best Model: Logistic Regression")
        best_model = LR[0]
        best_model_name = "Logistic Regression"
    elif best_model_accuracy == svc[1]:
        print("Best Model: SVC")
        best_model = svc[0]
        best_model_name = "SVC"
    elif best_model_accuracy == sgd[1]:
        print("Best Model: SGD")
        best_model = sgd[0]
        best_model_name = "SGD"
    else:
        print("No Best Model")

    # save model
    if save:
        try:
            dump(best_model, open(model_path + "model_classes_all.pkl", "wb"))
        except:
            print("Model not saved, Try adding the full path to the model")
    print("Model Saved")

    # remove log.txt if exists
    if os.path.exists("log_2.txt"):
        os.remove("log_2.txt")

    # log.txt
    with open("log_2.txt", "a") as f:
        f.write("Naive Bayes Accuracy: " + str(nb[1]) + "\n")
        f.write("KNN Accuracy: " + str(knn[1]) + "\n")
        f.write("Random Forest Accuracy: " + str(RF[1]) + "\n")
        f.write("Logistic Regression Accuracy: " + str(LR[1]) + "\n")
        f.write("SVC Accuracy: " + str(svc[1]) + "\n")
        f.write("SGD Accuracy: " + str(sgd[1]) + "\n")
        f.write("\n===========================================\n\n")
        f.write("Best Model Accuracy: " + str(best_model_accuracy) + "\n")
        f.write("Best Model: " + str(best_model) + "\n")
        f.write("Best Model Name: " + str(best_model_name) + "\n")
        f.write("\n===========================================\n\n")
        f.write("Best Model Classification Report: \n" + str(
            classification_report(y_test, best_model.predict(X_test), target_names=tags)) + "\n")

    print("Training Completed")
    print("Model location: ", model_path + "model_classes.pkl")

    y_pred = best_model.predict(X_test)

    # plot graphs
    graph_plotter = GraphPlotter(X_train, y_train, X_test, y_test, y_pred, best_model, df, save_path="log_all/")

    graph_plotter.plot_confusion_matrix()
    graph_plotter.plot_roc_curve()
    graph_plotter.plot_learning_curve()
    graph_plotter.plot_feature_importance()

    return best_model_accuracy, best_model


def predict_classes(text, model=None):
    print([text])
    if model is None:
        model = load(open(model_path + "model_classes_all.pkl", "rb"))

    pred = model.predict([text])

    # convert numeric to class   Sports, Religious, Political, Sexual, Education, and Entertainment.
    num_to_class = {
        "class": {
            0: "Sports",
            1: "Religious",
            2: "Political",
            3: "Sexual",
            4: "Education",
            5: "Entertainment"
        }
    }

    if pred[0] not in num_to_class["class"]:
        print("Class not found")
        return None
    else:
        pred = num_to_class["class"][pred[0]]
        print("Predicted Class: ", pred)

    return pred


def processing_audio_classes(df, wav2vec_model):
    # predict hate for each unique text and add that to all rows with same text

    # get unique text
    unique_text = df["text"].unique()

    # predict hate for each unique text
    for text in unique_text:
        # get all rows with same text
        rows = df[df["text"] == text]

        # get first row
        row = rows.iloc[0]

        # get classes
        hate = predict_classes(row["text"])

        # add hate to all rows with same text
        df.loc[df["text"] == text, "classes"] = hate

    return df


# train model
ac, model = train(save=True)

# predict
text = " අනේ පල හුත්තො තොපේ මහ එකාගෙ දවසක වියඳම කීයද ඒ දවස් වල"
predict_classes(text, model)
