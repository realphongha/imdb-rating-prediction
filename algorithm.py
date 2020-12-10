import json
import math
import random
import preprocess
import evaluate
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression

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
    random_forest = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0).fit(X_train, y_train)
    pred = random_forest.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    return err

def knn(X_train, y_train, X_test, y_test):
    # knn:
    k = int(math.sqrt(len(X_train[0])))
    k = k if k & 1 else k + 1 # k will be odd
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    return err

def naive_bayes(X_train, y_train, X_test, y_test):
    # naive bayes:
    nb = GaussianNB().fit(X_train, y_train)
    pred = nb.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    return err

def decision_tree_reg(X_train, y_train, X_test, y_test):
    # decision tree:
    d_tree = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
    pred = d_tree.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    return err

def logistic_reg(X_train, y_train, X_test, y_test):
    # logistic regression
    lg = LogisticRegression(random_state=0, solver='newton-cg', multi_class='ovr', max_iter=200, penalty='l2')
    lg.fit(X_train, y_train)
    pred = lg.predict(X_test)
    err = evaluate.error_margin(pred, y_test)/10.0
    return err

print("svr error margin:", svm_reg(X_train, y_train, X_test, y_test))
print("svr error margin:", svm_reg(X_train2, y_train2, X_test2, y_test2))

print("svc error margin:", svm_clf(X_train, y_train, X_test, y_test))
print("svc error margin:", svm_clf(X_train2, y_train2, X_test2, y_test2))

print("random forest error margin:", random_forest_reg(X_train, y_train, X_test, y_test))
print("random forest error margin:", random_forest_reg(X_train2, y_train2, X_test2, y_test2))

print("knn clf error margin:", knn(X_train, y_train, X_test, y_test))
print("knn clf error margin:", knn(X_train2, y_train2, X_test2, y_test2))

print("GaussianNB error margin:", naive_bayes(X_train, y_train, X_test, y_test))
print("GaussianNB error margin:", naive_bayes(X_train2, y_train2, X_test2, y_test2))

print("decision tree error margin:", decision_tree_reg(X_train, y_train, X_test, y_test))
print("decision tree error margin:", decision_tree_reg(X_train2, y_train2, X_test2, y_test2))

print("logistic regression error margin:", logistic_reg(X_train, y_train, X_test, y_test))
print("logistic regression error margin:", logistic_reg(X_train2, y_train2, X_test2, y_test2))