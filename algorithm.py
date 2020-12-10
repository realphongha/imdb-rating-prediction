import json
import math
import random
import numpy as np
import preprocess
import evaluate

from time import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn import metrics
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

TRAIN_FILE = "crawler/data1.json"
EXAMPLE_FILE = "example.json"
PREDICTION_FILE = "prediction.txt"
TRAIN_TEST_RATION = 0.8


train_file = open(TRAIN_FILE, "r")
raw_data = []
lines = train_file.read().splitlines()
random.shuffle(lines)
for line in lines:
    raw_data.append(json.loads(line))
data_train, data_test = preprocess.split_data(raw_data, TRAIN_TEST_RATION)
# imdb rating as float:
X_train, y_train, names_train, X_test, y_test, names_test = preprocess.preprocess(data_train, data_test, False)
# rounded imdb rating:
X_train2, y_train2, names_train2, X_test2, y_test2, names_test2 = preprocess.preprocess(data_train, data_test, True)

# example_file = open(EXAMPLE_FILE, "r")
# raw_data = []
# for line in example_file.read().splitlines():
#     raw_data.append(json.loads(line))
# X_example, y_example, names_example = preprocess.preprocess(raw_data, True)

def svm_reg(X_train, y_train, X_test, y_test):
    # svr
    svr = make_pipeline(StandardScaler(), SVR(gamma="scale", C=1.1, epsilon=1.1))
    svr.fit(X_train, y_train)
    pred = svr.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    # print(svr.get_params())
    # print(svr.predict(X_example))
    # file = open(PREDICTION_FILE, "w")
    # for i in range(len(pred)):
    #     file.write(names_test[i] + ": " + str(pred[i]) + "\n")
    # file.close()
    return err

def svm_clf(X_train, y_train, X_test, y_test):
    # svc
    svc = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    svc.fit(X_train, y_train)
    pred = svc.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    return err

def random_forest_reg(X_train, y_train, X_test, y_test):
    # random forest reg:
    random_forest = RandomForestRegressor(n_estimators=100, min_samples_split=2, min_samples_leaf=2,
                                          max_features='auto', max_depth=40, bootstrap=True, n_jobs=-1)
    random_forest.fit(X_train, y_train)
    pred = random_forest.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    return err

def knn(X_train, y_train, X_test, y_test):
    # knn:
    # k = int(math.sqrt(len(X_train[0])))
    # k = k if k & 1 else k + 1 # k will be odd
    knn = KNeighborsClassifier(algorithm='auto', n_jobs=-1, n_neighbors=11, weights='distance')
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    return err

def naive_bayes(X_train, y_train, X_test, y_test):
    # naive bayes:
    nb = GaussianNB().fit(X_train, y_train)
    pred = nb.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    # print(nb.get_params())
    return err

def decision_tree_reg(X_train, y_train, X_test, y_test):
    # decision tree:
    d_tree = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
    pred = d_tree.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    return err

def logistic_reg(X_train, y_train, X_test, y_test):
    # logistic regression
    lg = LogisticRegression(random_state=0, solver='newton-cg', multi_class='ovr', max_iter=200, penalty='l2',
                            n_jobs=-1)
    lg.fit(X_train, y_train)
    pred = lg.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    return err

def knn_op_params(X_train, y_train, X_test, y_test):
    # knn:
    parameters = {'weights': ['distance', 'uniform'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}
    scoring = {'err_margin': metrics.make_scorer(evaluate.error_margin, greater_is_better=False)}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters, scoring=scoring, refit='err_margin', n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    print("Best params:", clf.best_params_)
    return err

def random_forest_reg_op_params(X_train, y_train, X_test, y_test):
    # random forest reg:
    # parameters = {'bootstrap': [True, False],
    #               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
    #               'max_features': ['auto', 'sqrt'],
    #               'min_samples_leaf': [1, 2, 4],
    #               'min_samples_split': [2, 5, 10],
    #               'n_estimators': [100, 150, 200]}
    parameters = {"n_estimators": [100, 300, 500, 800, 1200],
                  "max_depth": [5, 8, 15, 25, 30],
                  "min_samples_split": [2, 5, 10, 15, 100],
                  "min_samples_leaf": [1, 2, 5, 10]}
    random_forest = RandomForestRegressor()
    scoring = {'err_margin': metrics.make_scorer(evaluate.error_margin, greater_is_better=False)}
    # clf = RandomizedSearchCV(estimator=random_forest, param_distributions=parameters, n_iter=20, cv=3, verbose=2,
    #                          random_state=42, n_jobs=-1, scoring=scoring, refit='err_margin')
    clf = GridSearchCV(estimator=random_forest, param_grid=parameters, verbose=2, n_jobs=2,
                       scoring=scoring, refit='err_margin')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    print("Best params:", clf.best_params_)
    return err

def svm_reg_op_params(X_train, y_train, X_test, y_test):
    # svm reg:
    # parameters = {"kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    #               "gamma": ['scale', 'auto'],
    #               "C": [x for x in np.linspace(start=1.0, stop=10.0, num=100, dtype=float)],
    #               "epsilon": [x for x in np.linspace(start=0.1, stop=0.9, num=9, dtype=float)],
    #               "shrinking": [True, False]}
    parameters = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
    svr = SVR()
    scoring = {'err_margin': metrics.make_scorer(evaluate.error_margin, greater_is_better=False)}
    # clf = RandomizedSearchCV(estimator=svr, param_distributions=parameters, n_iter=20, cv=3, verbose=2,
    #                          random_state=1, n_jobs=2, scoring=scoring, refit='err_margin')
    clf = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=2, scoring=scoring, refit='err_margin', verbose=2)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    print("Best params:", clf.best_params_)
    return err

def svm_clf_op_params(X_train, y_train, X_test, y_test):
    # svm clf:
    # parameters = {"kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    #               "gamma": ['scale', 'auto'],
    #               "C": [x for x in np.linspace(start=1.0, stop=10.0, num=100, dtype=float)],
    #               "shrinking": [True, False]}
    parameters = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
    svc = SVC()
    scoring = {'err_margin': metrics.make_scorer(evaluate.error_margin, greater_is_better=False)}
    # clf = RandomizedSearchCV(estimator=svc, param_distributions=parameters, n_iter=20, cv=3, verbose=2,
    #                          random_state=1, n_jobs=2, scoring=scoring, refit='err_margin')
    clf = GridSearchCV(estimator=svc, param_grid=parameters, n_jobs=2, scoring=scoring, refit='err_margin',
                       verbose=2)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    print("Best params:", clf.best_params_)
    return err

def logistic_reg_op_params(X_train, y_train, X_test, y_test):
    # svm clf:
    parameters = {"solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  "multi_class": ['auto', 'ovr', 'multinomial'],
                  "penalty": ['l1', 'l2', 'elasticnet', 'none'],
                  "warm_start": [True, False],
                  "C": [x for x in np.linspace(start=1.0, stop=10.0, num=100, dtype=float)]}
    lg = LogisticRegression( max_iter=200, penalty='l2',
                            n_jobs=-1)
    scoring = {'err_margin': metrics.make_scorer(evaluate.error_margin, greater_is_better=False)}
    clf = RandomizedSearchCV(estimator=lg, param_distributions=parameters, n_iter=20, cv=3, verbose=2,
                             random_state=1, n_jobs=2, scoring=scoring, refit='err_margin')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    print("Best params:", clf.best_params_)
    return err

print("svr error margin:", svm_reg(X_train, y_train, X_test, y_test))
print("svr error margin (rounded):", svm_reg(X_train2, y_train2, X_test2, y_test2))
print("svr error margin (optimized params):", svm_reg_op_params(X_train, y_train, X_test, y_test))
#
print("svc error margin:", svm_clf(X_train, y_train, X_test, y_test))
print("svc error margin (rounded):", svm_clf(X_train2, y_train2, X_test2, y_test2))
print("svc error margin (optimized params):", svm_clf_op_params(X_train, y_train, X_test, y_test))
#
print("random forest error margin:", random_forest_reg(X_train, y_train, X_test, y_test))
print("random forest error margin (rounded):", random_forest_reg(X_train2, y_train2, X_test2, y_test2))
print("random forest error margin (optimized params):", random_forest_reg_op_params(X_train, y_train, X_test, y_test))
#
# print("knn clf error margin:", knn(X_train, y_train, X_test, y_test))
# print("knn clf error margin (rounded):", knn(X_train2, y_train2, X_test2, y_test2))
# print("knn clf error margin (optimized params):", knn_op_params(X_train, y_train, X_test, y_test))
#
# print("GaussianNB error margin:", naive_bayes(X_train, y_train, X_test, y_test))
# print("GaussianNB error margin (rounded):", naive_bayes(X_train2, y_train2, X_test2, y_test2))
#
# print("decision tree error margin:", decision_tree_reg(X_train, y_train, X_test, y_test))
# print("decision tree error margin (rounded):", decision_tree_reg(X_train2, y_train2, X_test2, y_test2))

# print("logistic regression error margin:", logistic_reg(X_train, y_train, X_test, y_test))
# print("logistic regression error margin (rounded):", logistic_reg(X_train2, y_train2, X_test2, y_test2))
# print("logistic regression error margin (optimized params):", logistic_reg_op_params(X_train, y_train, X_test, y_test))