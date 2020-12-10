import datetime
from sklearn.preprocessing import scale

# RATING_SYSTEM = {"G": 0.0, "PG": 10.0, "PG-13": 20.0, "R": 30.0, "Unrated": 40.0, "Not Rated": 50.0}
RATING_SYSTEM = {"G": 2.0, "PG": 4.0, "PG-13": 6.0, "R": 8.0, "Unrated": 10.0, "Not Rated": 10.0}
CURRENT_YEAR = datetime.datetime.now().year


def word_encode(word):
    assert type(word) == str, str(word) + " is not a string!"
    encoded_word = 0
    for c in word:
        encoded_word += ord(c)
    return float(encoded_word)


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


def split_data(data, ration):
    new_data = []
    for line in data:
        if int(line["year"]) == CURRENT_YEAR:
            continue
            # line["year"] = str(int(line["year"])*2)
        if line["rated"] not in RATING_SYSTEM:
            continue
        new_data.append(line)
    train_number = int(len(new_data) * ration)
    data_train = new_data[:train_number]
    data_test = new_data[train_number:]
    return data_train, data_test


def preprocess(data_train, data_test, rounded=True):
    genres_score = {}
    actors_score = {}
    keywords_score = {}
    directors_score = {}
    writers_score = {}
    for line in data_train:
        rating = int(float(line["imdb_rating"])) if rounded else float(line["imdb_rating"])
        genres = line["genres"].split(",")
        for genre in genres:
            if genre in genres_score:
                genres_score[genre][0] += rating
                genres_score[genre][1] += 1
            else:
                genres_score[genre] = [rating, 1]

        actors = line["actors"].split(",")
        for actor in actors:
            if actor in actors_score:
                actors_score[actor][0] += rating
                actors_score[actor][1] += 1
            else:
                actors_score[actor] = [rating, 1]

        keywords = line["keywords"].split(",")
        for keyword in keywords:
            if keyword in keywords_score:
                keywords_score[keyword][0] += rating
                keywords_score[keyword][1] += 1
            else:
                keywords_score[keyword] = [rating, 1]

        keywords = line["keywords"].split(",")
        for keyword in keywords:
            if keyword in keywords_score:
                keywords_score[keyword][0] += rating
                keywords_score[keyword][1] += 1
            else:
                keywords_score[keyword] = [rating, 1]

        directors = line["directors"].split(",") if type(line["directors"]) == str else line["directors"]
        for director in directors:
            if director in directors_score:
                directors_score[director][0] += rating
                directors_score[director][1] += 1
            else:
                directors_score[director] = [rating, 1]

        writers = line["writers"].split(",")
        for writer in writers:
            if writer in writers_score:
                writers_score[writer][0] += rating
                writers_score[writer][1] += 1
            else:
                writers_score[writer] = [rating, 1]

    X_train = []
    y_train = []
    names_train = []
    for line in data_train:

        X_line = []

        X_line.append(float(CURRENT_YEAR-int(line["year"])) / 10.0)

        X_line.append(RATING_SYSTEM[line["rated"]])

        X_line.append(time_str_to_num(line["runtime"])/10.0)

        X_line.append(5.0 if line["awards_oscar"] != None else 0.0)

        X_line.append(float(line["imdb_votes"])/1000.0)

        X_line.append(float(line["metascore"])/10.0 if line["metascore"] != None else 5.0)

        genres = line["genres"].split(",")
        # for i in range(3):
        #     try:
        #         X_line.append(word_encode(genres[i]))
        #     except IndexError:
        #         X_line.append(0.0)
        genres_sum_score = 0.0
        for genre in genres:
            genres_sum_score += genres_score[genre][0]/genres_score[genre][1]
        X_line.append(genres_sum_score/len(genres))

        actors = line["actors"].split(",")
        # for i in range(3):
        #     try:
        #         X_line.append(word_encode(actors[i]))
        #     except IndexError:
        #         X_line.append(0.0)
        actors_sum_score = 0.0
        for actor in actors:
            actors_sum_score += actors_score[actor][0]/actors_score[actor][1]
        X_line.append(actors_sum_score / len(actors))

        keywords = line["keywords"].split(",")
        # for i in range(3):
        #     try:
        #         X_line.append(word_encode(keywords[i]))
        #     except IndexError:
        #         X_line.append(0.0)
        keywords_sum_score = 0.0
        for keyword in keywords:
            keywords_sum_score += keywords_score[keyword][0]/keywords_score[keyword][1]
        X_line.append(keywords_sum_score / len(keywords))

        directors = line["directors"].split(",") if type(line["directors"]) == str else line["directors"]
        # X_line.append(word_encode(directors[0]))
        directors_sum_score = 0.0
        for director in directors:
            directors_sum_score += directors_score[director][0] / directors_score[director][1]
        X_line.append(directors_sum_score / len(directors))

        writers = line["writers"].split(",")
        # X_line.append(word_encode(writers[0]))
        writers_sum_score = 0.0
        for writer in writers:
            writers_sum_score += writers_score[writer][0] / writers_score[writer][1]
        X_line.append(writers_sum_score / len(writers))

        y_line = int(float(line["imdb_rating"]))*10 if rounded else int(float(line["imdb_rating"])*10)
        name = line["title"] + " - " + line["year"]

        X_train.append(X_line)
        y_train.append(y_line)
        names_train.append(name)

    X_test = []
    y_test = []
    names_test = []
    for line in data_test:

        X_line = []

        X_line.append(float(CURRENT_YEAR-int(line["year"])) / 10.0)

        X_line.append(RATING_SYSTEM[line["rated"]])

        X_line.append(time_str_to_num(line["runtime"])/10.0)

        X_line.append(5.0 if line["awards_oscar"] != None else 0.0)

        X_line.append(float(line["imdb_votes"])/1000.0)

        X_line.append(float(line["metascore"])/10.0 if line["metascore"] != None else 5.0)

        genres = line["genres"].split(",")
        # for i in range(3):
        #     try:
        #         X_line.append(word_encode(genres[i]))
        #     except IndexError:
        #         X_line.append(0.0)
        genres_sum_score = 0.0
        for genre in genres:
            try:
                genres_sum_score += genres_score[genre][0] / genres_score[genre][1]
            except KeyError:
                genres_sum_score += 5.0
        X_line.append(genres_sum_score / len(genres))

        actors = line["actors"].split(",")
        # for i in range(3):
        #     try:
        #         X_line.append(word_encode(actors[i]))
        #     except IndexError:
        #         X_line.append(0.0)
        actors_sum_score = 0.0
        for actor in actors:
            try:
                actors_sum_score += actors_score[actor][0] / actors_score[actor][1]
            except KeyError:
                actors_sum_score += 5.0
        X_line.append(actors_sum_score / len(actors))

        keywords = line["keywords"].split(",")
        # for i in range(3):
        #     try:
        #         X_line.append(word_encode(keywords[i]))
        #     except IndexError:
        #         X_line.append(0.0)
        keywords_sum_score = 0.0
        for keyword in keywords:
            try:
                keywords_sum_score += keywords_score[keyword][0] / keywords_score[keyword][1]
            except KeyError:
                keywords_sum_score += 5.0
        X_line.append(keywords_sum_score / len(keywords))

        directors = line["directors"].split(",") if type(line["directors"]) == str else line["directors"]
        # X_line.append(word_encode(directors[0]))
        directors_sum_score = 0.0
        for director in directors:
            try:
                directors_sum_score += directors_score[director][0] / directors_score[director][1]
            except KeyError:
                directors_sum_score += 5.0
        X_line.append(directors_sum_score / len(directors))

        writers = line["writers"].split(",")
        # X_line.append(word_encode(writers[0]))
        writers_sum_score = 0.0
        for writer in writers:
            try:
                writers_sum_score += writers_score[writer][0] / writers_score[writer][1]
            except KeyError:
                writers_sum_score += 5.0
        X_line.append(writers_sum_score / len(writers))

        y_line = int(float(line["imdb_rating"]))*10 if rounded else int(float(line["imdb_rating"])*10)
        name = line["title"] + " - " + line["year"]

        X_test.append(X_line)
        y_test.append(y_line)
        names_test.append(name)
    return scale(X_train), y_train, names_train, scale(X_test), y_test, names_test


if __name__ == "__main__":
    pass
