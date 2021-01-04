import json
import random
import preprocess
from models import *

TRAIN_FILE = "crawler/data3.json"
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

# to predict some example movies:
# example_file = open(EXAMPLE_FILE, "r")
# raw_data = []
# for line in example_file.read().splitlines():
#     raw_data.append(json.loads(line))
# X_example, y_example, names_example = preprocess.preprocess(raw_data, True)

print("svr error margin:", svm_reg(X_train, y_train, X_test, y_test))
print("svr error margin (rounded):", svm_reg(X_train2, y_train2, X_test2, y_test2))
# print("svr error margin (optimized params):", svm_reg_op_params(X_train, y_train, X_test, y_test))

print("svc error margin:", svm_clf(X_train, y_train, X_test, y_test))
print("svc error margin (rounded):", svm_clf(X_train2, y_train2, X_test2, y_test2))
# print("svc error margin (optimized params):", svm_clf_op_params(X_train, y_train, X_test, y_test))

print("random forest error margin:", random_forest_reg(X_train, y_train, X_test, y_test))
print("random forest error margin (rounded):", random_forest_reg(X_train2, y_train2, X_test2, y_test2))
# print("random forest error margin (optimized params):", random_forest_reg_op_params(X_train, y_train, X_test, y_test))

print("knn clf error margin:", knn(X_train, y_train, X_test, y_test))
print("knn clf error margin (rounded):", knn(X_train2, y_train2, X_test2, y_test2))
# print("knn clf error margin (optimized params):", knn_op_params(X_train, y_train, X_test, y_test))

print("GaussianNB error margin:", naive_bayes(X_train, y_train, X_test, y_test))
print("GaussianNB error margin (rounded):", naive_bayes(X_train2, y_train2, X_test2, y_test2))

print("logistic regression error margin:", logistic_reg(X_train, y_train, X_test, y_test))
print("logistic regression error margin (rounded):", logistic_reg(X_train2, y_train2, X_test2, y_test2))
# print("logistic regression error margin (optimized params):", logistic_reg_op_params(X_train, y_train, X_test, y_test))

print("ridge regression error margin:", ridge_reg(X_train, y_train, X_test, y_test))
print("ridge regression error margin (rounded):", ridge_reg(X_train2, y_train2, X_test2, y_test2))
# print("ridge regression error margin (optimized params):", ridge_reg_op_params(X_train, y_train, X_test, y_test))

print("ridge clf error margin:", ridge_clf(X_train, y_train, X_test, y_test))
print("ridge clf error margin (rounded):", ridge_clf(X_train2, y_train2, X_test2, y_test2))
# print("ridge clf error margin (optimized params):", ridge_clf_op_params(X_train, y_train, X_test, y_test))