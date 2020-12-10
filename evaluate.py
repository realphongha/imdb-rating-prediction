THRESHOLD_PREDICTION = 10

def error_margin(y_predict, y_test):
    assert len(y_predict) == len(y_test), "y_predict and y_test must have same length!"
    sum_error_margin = 0.0
    for i in range(len(y_predict)):
        sum_error_margin += abs(y_predict[i]-y_test[i])
    return sum_error_margin/len(y_predict)


def accuracy_score(y_predict, y_test, threshold_pred=THRESHOLD_PREDICTION):
    correct = []
    for i in range(len(y_test)):
        if y_test[i] >= y_predict[i] - threshold_pred and y_test[i] <= y_predict[i] + threshold_pred:
            correct.append(1)
        else:
            correct.append(0)
    return sum(map(int, correct)) / len(correct)