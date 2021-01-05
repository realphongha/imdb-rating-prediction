from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC

import evaluate


def svm_reg(X_train, y_train, X_test, y_test):
    # svr
    # svr = make_pipeline(StandardScaler(), SVR(gamma="scale", C=1.1, epsilon=1.1))
    svr = SVR(gamma="scale", C=1.1, epsilon=1.1)
    svr.fit(X_train, y_train)
    pred = svr.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    # print(svr.get_params())
    # print(svr.predict(X_example))
    # file = open(PREDICTION_FILE, "w")
    # for i in range(len(pred)):
    #     file.write(names_test[i] + ": " + str(pred[i]) + "\n")
    # file.close()
    return err, pred


def svm_clf(X_train, y_train, X_test, y_test):
    # svc
    # svc = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    svc = SVC(gamma="auto")
    svc.fit(X_train, y_train)
    pred = svc.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    return err, pred


def random_forest_reg(X_train, y_train, X_test, y_test):
    # random forest reg:
    random_forest = RandomForestRegressor(n_estimators=100, min_samples_split=5, min_samples_leaf=2)
    random_forest.fit(X_train, y_train)
    pred = random_forest.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    return err, pred


def knn(X_train, y_train, X_test, y_test):
    # knn:
    # k = int(math.sqrt(len(X_train[0])))
    # k = k if k & 1 else k + 1 # k will be odd
    knn = KNeighborsClassifier(algorithm='auto', n_jobs=-1, n_neighbors=15, weights='distance', metric='euclidean')
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    return err, pred


def naive_bayes(X_train, y_train, X_test, y_test):
    # naive bayes:
    nb = GaussianNB().fit(X_train, y_train)
    pred = nb.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    # print(nb.get_params())
    return err, pred


def logistic_reg(X_train, y_train, X_test, y_test):
    # logistic regression
    lg = LogisticRegression(random_state=0, solver='newton-cg', multi_class='ovr', max_iter=200, penalty='l2',
                            n_jobs=-1)
    lg.fit(X_train, y_train)
    pred = lg.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    return err, pred


def ridge_reg(X_train, y_train, X_test, y_test):
    ridge = Ridge(alpha=0.5, solver='sag')
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    return err, pred


def ridge_clf(X_train, y_train, X_test, y_test):
    ridge = RidgeClassifier(alpha=0.9, solver='lsqr')
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    return err, pred


def knn_op_params(X_train, y_train, X_test, y_test):
    parameters = {'weights': ['distance', 'uniform'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
                  'metric': ['euclidean']}
    scoring = {'err_margin': metrics.make_scorer(evaluate.error_margin, greater_is_better=False)}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters, scoring=scoring, refit='err_margin', n_jobs=-1, verbose=2)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    print("Best params:", clf.best_params_)
    return err


def random_forest_reg_op_params(X_train, y_train, X_test, y_test):
    # parameters = {'bootstrap': [True, False],
    #               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
    #               'max_features': ['auto', 'sqrt'],
    #               'min_samples_leaf': [1, 2, 4],
    #               'min_samples_split': [2, 5, 10],
    #               'n_estimators': [100, 150, 200]}
    parameters = {"n_estimators": [10, 100, 500, 1000],
                  "max_depth": [5, 8, 15, 25, 30],
                  "min_samples_split": [2, 5, 10, 15, 100],
                  "min_samples_leaf": [1, 2, 5, 10],
                  "max_features": list(range(1, len(X_train[0]) // 2 + 1))}
    random_forest = RandomForestRegressor()
    scoring = {'err_margin': metrics.make_scorer(evaluate.error_margin, greater_is_better=False)}
    clf = RandomizedSearchCV(estimator=random_forest, param_distributions=parameters, n_iter=20, cv=3, verbose=2,
                             random_state=42, n_jobs=-1, scoring=scoring, refit='err_margin')
    # clf = GridSearchCV(estimator=random_forest, param_grid=parameters, verbose=2, n_jobs=2,
    #                    scoring=scoring, refit='err_margin')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    print("Best params:", clf.best_params_)
    return err


def svm_reg_op_params(X_train, y_train, X_test, y_test):
    # parameters = {"kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    #               "gamma": ['scale', 'auto'],
    #               "C": [x for x in np.linspace(start=1.0, stop=10.0, num=100, dtype=float)],
    #               "epsilon": [x for x in np.linspace(start=0.1, stop=0.9, num=9, dtype=float)],
    #               "shrinking": [True, False]}
    parameters = {'C': [0.1, 1, 10, 100], 'gamma': ['auto', 'scale'], 'kernel': ['rbf']}
    svr = SVR()
    scoring = {'err_margin': metrics.make_scorer(evaluate.error_margin, greater_is_better=False)}
    # clf = RandomizedSearchCV(estimator=svr, param_distributions=parameters, n_iter=20, cv=3, verbose=2,
    #                          random_state=1, n_jobs=2, scoring=scoring, refit='err_margin')
    clf = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=2, scoring=scoring, refit='err_margin', verbose=2)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    print("Best params:", clf.best_params_)
    return err


def svm_clf_op_params(X_train, y_train, X_test, y_test):
    # parameters = {"kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    #               "gamma": ['scale', 'auto'],
    #               "C": [x for x in np.linspace(start=1.0, stop=10.0, num=100, dtype=float)],
    #               "shrinking": [True, False]}
    parameters = {'C': [0.1, 1, 10, 100], 'gamma': ['auto', 'scale'], 'kernel': ['rbf']}
    svc = SVC()
    scoring = {'err_margin': metrics.make_scorer(evaluate.error_margin, greater_is_better=False)}
    # clf = RandomizedSearchCV(estimator=svc, param_distributions=parameters, n_iter=20, cv=3, verbose=2,
    #                          random_state=1, n_jobs=2, scoring=scoring, refit='err_margin')
    clf = GridSearchCV(estimator=svc, param_grid=parameters, n_jobs=2, scoring=scoring, refit='err_margin',
                       verbose=2)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    print("Best params:", clf.best_params_)
    return err


def logistic_reg_op_params(X_train, y_train, X_test, y_test):
    parameters = {"solver": ['newton-cg', 'lbfgs', 'liblinear'],
                  "multi_class": ['auto'],
                  "penalty": ['l2'],
                  "C": [0.01, 0.1, 1, 10, 100]}
    lg = LogisticRegression()
    scoring = {'err_margin': metrics.make_scorer(evaluate.error_margin, greater_is_better=False)}
    # clf = RandomizedSearchCV(estimator=lg, param_distributions=parameters, n_iter=50, cv=3, verbose=2,
    #                          random_state=1, n_jobs=2, scoring=scoring, refit='err_margin')
    clf = GridSearchCV(estimator=lg, param_grid=parameters, verbose=2, n_jobs=2, scoring=scoring, refit='err_margin')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    print("Best params:", clf.best_params_)
    return err


def ridge_reg_op_params(X_train, y_train, X_test, y_test):
    parameters = {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                  }
    ridge = Ridge()
    scoring = {'err_margin': metrics.make_scorer(evaluate.error_margin, greater_is_better=False)}

    clf = GridSearchCV(estimator=ridge, param_grid=parameters, verbose=2, n_jobs=2, scoring=scoring,
                       refit='err_margin')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    print("Best params:", clf.best_params_)
    return err


def ridge_clf_op_params(X_train, y_train, X_test, y_test):
    parameters = {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                  }
    ridge = RidgeClassifier()
    scoring = {'err_margin': metrics.make_scorer(evaluate.error_margin, greater_is_better=False)}

    clf = GridSearchCV(estimator=ridge, param_grid=parameters, verbose=2, n_jobs=2, scoring=scoring,
                       refit='err_margin')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    err = evaluate.error_margin(pred, y_test) / 10.0
    print("Best params:", clf.best_params_)
    return err
