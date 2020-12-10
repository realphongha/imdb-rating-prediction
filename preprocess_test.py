import json
import math
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
FILE = "crawler/data1.json"
RATING_SYSTEM = {"G": 0.0, "PG": 10.0, "PG-13": 20.0, "R": 30.0, "Unrated": 40.0, "Not Rated": 50.0}
TRAIN_TEST_RATION = 0.8


def time_str_to_num(time):
    if time[:2] == "PT":
        h = 0
        m = 0
        if "H" in time:
            h = int(time[2:time.index("H")])
            if "M" in time:
                m = int(time[time.index("H")+1:-1])
        else:
            m = int(time[2:-1])
        return h*60 + m
    else:
        return None


file = open(FILE, "r")
X = []
Y = []
for line in file.read().splitlines():
    movie = json.loads(line)
    try:
        if movie["year"] == "2020":
            continue
        # X.append([(int(movie["year"])-1900)/10.0, RATING_SYSTEM[movie["rated"]], time_str_to_num(movie["runtime"]),
        #           int(movie["imdb_votes"])/10, int(movie["metascore"])*2])
        X.append([(int(movie["year"])-1900)/10, RATING_SYSTEM[movie["rated"]], int(movie["imdb_votes"]),
                  int(movie["metascore"]) * 10, time_str_to_num(movie["runtime"])/2,
                  50 if movie["awards_oscar"] != None else 0])
        # X.append([int(movie["imdb_votes"]),
        #           int(movie["metascore"]) * 10,
        #           100 if movie["awards_oscar"] != None else 0])
        Y.append(round(float(movie["imdb_rating"])))
        # Y.append(float(movie["imdb_rating"]))
    except:
        pass

# print(Y)
# print(len(X))

train_number = int(len(X)*0.8)
X_train = X[:train_number]
y_train = Y[:train_number]
X_test = X[train_number:]
y_test = Y[train_number:]

# # linear reg:
# lin_reg = LinearRegression().fit(X_train, y_train)
# print("lin reg score:", lin_reg.score(X_test, y_test))
#
# # random forest reg:
# random_forest = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0).fit(X_train, y_train)
# print("random forest score:", random_forest.score(X_test, y_test))
#
# # knn clf:
# k = int(math.sqrt(len(X_train[0])))
# k = k if k & 1 else k + 1 # k will be odd
# knn = KNeighborsClassifier(n_neighbors=k)
# knn.fit(X_train, y_train)
# pred = knn.predict(X_test)
# acc = metrics.accuracy_score(y_test, pred)
# print("knn clf accuracy score:", acc)
#
# # svm:
# svm = make_pipeline(StandardScaler(), SVC(gamma="auto"))
# # svm = SVC(gamma="auto")
# svm.fit(X_train, y_train)
# pred = svm.predict(X_test)
# print(X_test)
# print(pred)
# acc = metrics.accuracy_score(y_test, pred)
# print("svm accuracy score:", acc)

# best_c = None
# best_e = None
# best_score = -1
# for c0 in range(1, 50, 1):
#     for e0 in range(1, 50, 1):
#         # sgd = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
#         sgd = make_pipeline(StandardScaler(), SVR(C=c0/10.0, epsilon=e0/10.0))
#         sgd.fit(X_train, y_train)
#         pred = sgd.predict(X_test)
#         pred = [round(x) for x in pred]
#         acc = metrics.accuracy_score(y_test, pred)
#         print(c0, e0)
#         print("sgd accuracy score:", acc)
#         if acc > best_score:
#             best_score = acc
#             best_c = c0
#             best_e = e0
# print(best_c, best_e, best_score)

sgd = make_pipeline(StandardScaler(), SVR(gamma="auto", C=1.1, epsilon=1.1))
# sgd = SVR(gamma="scale", C=1.1, epsilon=1.1)
sgd.fit(X_train, y_train)
pred = sgd.predict(X_test)
pred = [round(x) for x in pred]
acc = metrics.accuracy_score(y_test, pred)
print("sgd accuracy score:", acc)
# print("sgd accuracy score:", sgd.score(X_test, y_test))