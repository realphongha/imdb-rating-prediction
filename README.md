# imdb-rating-prediction
Small program that crawls movies directly from imdb.com (or using API from http://www.omdbapi.com), preprocesses the data and trains ML models to predict movies' rating.

# requirements:
python 3 and sklearn, scrapy

# run crawler:
"scrapy crawl imdb" in "crawler" folder to start crawling.

# change Line 152 in run.py:
run_models() to run machine learning models once on train set and test set
run_experiment() to run models 20 times and get the result
run_optimize_params() to optimize machine learning models' parameters
run_demo() to run demo on some example data records (put them in example.json)
("python run.py" to execute)

# jupyter notebook file "imdb_rating_analysis.ipynb" -> using for statistics and visualizing.
