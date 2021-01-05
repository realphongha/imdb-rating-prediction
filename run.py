import json
import random
import preprocess
from models import *

TRAIN_FILE = "crawler/data3.json"
EXAMPLE_FILE = "example.json"
PREDICTION_FILE = "prediction.txt"
TRAIN_TEST_RATION = 0.8
ITER = 20

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


def divided_by_ten(x):
    return [i/10 for i in x]


def run_models():
    print("svr error margin:", svm_reg(X_train, y_train, X_test, y_test)[0])
    print("svr error margin (rounded):", svm_reg(X_train2, y_train2, X_test2, y_test2)[0])

    print("svc error margin:", svm_clf(X_train, y_train, X_test, y_test)[0])
    print("svc error margin (rounded):", svm_clf(X_train2, y_train2, X_test2, y_test2)[0])

    print("random forest error margin:", random_forest_reg(X_train, y_train, X_test, y_test)[0])
    print("random forest error margin (rounded):", random_forest_reg(X_train2, y_train2, X_test2, y_test2)[0])

    print("knn clf error margin:", knn(X_train, y_train, X_test, y_test)[0])
    print("knn clf error margin (rounded):", knn(X_train2, y_train2, X_test2, y_test2)[0])

    print("GaussianNB error margin:", naive_bayes(X_train, y_train, X_test, y_test)[0])
    print("GaussianNB error margin (rounded):", naive_bayes(X_train2, y_train2, X_test2, y_test2)[0])

    print("logistic regression error margin:", logistic_reg(X_train, y_train, X_test, y_test)[0])
    print("logistic regression error margin (rounded):", logistic_reg(X_train2, y_train2, X_test2, y_test2)[0])

    print("ridge regression error margin:", ridge_reg(X_train, y_train, X_test, y_test)[0])
    print("ridge regression error margin (rounded):", ridge_reg(X_train2, y_train2, X_test2, y_test2)[0])

    print("ridge clf error margin:", ridge_clf(X_train, y_train, X_test, y_test)[0])
    print("ridge clf error margin (rounded):", ridge_clf(X_train2, y_train2, X_test2, y_test2)[0])


def run_demo():
    # to predict some example movies:
    example_file = open(EXAMPLE_FILE, "r")
    raw_data = []
    for line in example_file.read().splitlines():
        raw_data.append(json.loads(line))
    raw_data, _ = preprocess.split_data(raw_data, 1.0)
    _, __, ___, X_example, y_example, names_example = preprocess.preprocess(data_train, raw_data, False)
    _, __, ___, X_example2, y_example2, names_example2 = preprocess.preprocess(data_train, raw_data, True)


    print("Prediction of movies:", names_example, "using KNN.")
    print("Real IMDB score:", divided_by_ten(y_example))
    # print("and rounded score:", divided_by_ten(y_example2))

    # print("svr error margin:", svm_reg(X_train, y_train, X_example, y_example)[1])
    # print("svr error margin (rounded):", svm_reg(X_train2, y_train2, X_test2, y_test2))

    # print("svc error margin:", svm_clf(X_train, y_train, X_test, y_test))
    # print("svc error margin (rounded):", svm_clf(X_train2, y_train2, X_test2, y_test2))

    # print("random forest error margin:", random_forest_reg(X_train, y_train, X_test, y_test))
    # print("random forest error margin (rounded):", random_forest_reg(X_train2, y_train2, X_test2, y_test2))

    print("KNN prediction score:", divided_by_ten(knn(X_train, y_train, X_example, y_example)[1]))
    # print("knn prediction: (rounded):", divided_by_ten(knn(X_train2, y_train2, X_example2, y_example2)[1]))

    # print("GaussianNB error margin:", naive_bayes(X_train, y_train, X_test, y_test))
    # print("GaussianNB error margin (rounded):", naive_bayes(X_train2, y_train2, X_test2, y_test2))

    # print("logistic regression error margin:", logistic_reg(X_train, y_train, X_test, y_test))
    # print("logistic regression error margin (rounded):", logistic_reg(X_train2, y_train2, X_test2, y_test2))

    # print("ridge regression error margin:", ridge_reg(X_train, y_train, X_test, y_test))
    # print("ridge regression error margin (rounded):", ridge_reg(X_train2, y_train2, X_test2, y_test2))

    # print("ridge clf error margin:", ridge_clf(X_train, y_train, X_test, y_test))
    # print("ridge clf error margin (rounded):", ridge_clf(X_train2, y_train2, X_test2, y_test2))


def run_optimize_params():
    print("svr error margin (optimized params):", svm_reg_op_params(X_train, y_train, X_test, y_test))
    print("svc error margin (optimized params):", svm_clf_op_params(X_train, y_train, X_test, y_test))
    print("random forest error margin (optimized params):", random_forest_reg_op_params(X_train, y_train, X_test, y_test))
    print("knn clf error margin (optimized params):", knn_op_params(X_train, y_train, X_test, y_test))
    print("logistic regression error margin (optimized params):", logistic_reg_op_params(X_train, y_train, X_test, y_test))
    print("ridge regression error margin (optimized params):", ridge_reg_op_params(X_train, y_train, X_test, y_test))
    print("ridge clf error margin (optimized params):", ridge_clf_op_params(X_train, y_train, X_test, y_test))


def run_experiment():
    scores = {"svr1": [], "svr2": [], "svc1": [], "svc2": [], "rf1": [], "rf2": [],
              "knn1": [], "knn2": [], "nb1": [], "nb2": [],
              "log1": [], "log2": [], "ridge_reg1": [], "ridge_reg2": [],
              "ridge_clf1": [], "ridge_clf2": []}
    for i in range(ITER):
        print("Iter:", i+1)
        raw_data = []
        random.shuffle(lines)
        for line in lines:
            raw_data.append(json.loads(line))
        data_train, data_test = preprocess.split_data(raw_data, TRAIN_TEST_RATION)
        # imdb rating as float:
        X_train, y_train, names_train, X_test, y_test, names_test = preprocess.preprocess(data_train, data_test, False)
        # rounded imdb rating:
        X_train2, y_train2, names_train2, X_test2, y_test2, names_test2 = preprocess.preprocess(data_train, data_test, True)

        scores["svr1"].append(svm_reg(X_train, y_train, X_test, y_test))
        scores["svr2"].append(svm_reg(X_train2, y_train2, X_test2, y_test2))
        scores["svc1"].append(svm_clf(X_train, y_train, X_test, y_test))
        scores["svc2"].append(svm_clf(X_train2, y_train2, X_test2, y_test2))
        scores["rf1"].append(random_forest_reg(X_train, y_train, X_test, y_test))
        scores["rf2"].append(random_forest_reg(X_train2, y_train2, X_test2, y_test2))
        scores["knn1"].append(knn(X_train, y_train, X_test, y_test))
        scores["knn2"].append(knn(X_train2, y_train2, X_test2, y_test2))
        scores["nb1"].append(naive_bayes(X_train, y_train, X_test, y_test))
        scores["nb2"].append(naive_bayes(X_train2, y_train2, X_test2, y_test2))
        scores["log1"].append(logistic_reg(X_train, y_train, X_test, y_test))
        scores["log2"].append(logistic_reg(X_train2, y_train2, X_test2, y_test2))
        scores["ridge_reg1"].append(ridge_reg(X_train, y_train, X_test, y_test))
        scores["ridge_reg2"].append(ridge_reg(X_train2, y_train2, X_test2, y_test2))
        scores["ridge_clf1"].append(ridge_clf(X_train, y_train, X_test, y_test))
        scores["ridge_clf2"].append(ridge_clf(X_train2, y_train2, X_test2, y_test2))

    for key in scores:
        mean = sum(scores[key])/len(scores[key])
        scores[key].append(mean)
        print(key, mean)

    print(scores)
    return scores

if __name__ == "__main__":
    run_demo()