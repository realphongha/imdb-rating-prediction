"""
Simple crawler to crawl movies' data from IMDB.
Phong Ha, October 2020
github.com/realphongha
"""

import json
from .constant import *


class MoviePipeline:

    def open_spider(self, spider):
        if CRAWL_TOOL == OUR_CRAWLER:
            self.movies = []
        else:
            self.movie_jsons = []

    def process_item(self, item, spider):
        if CRAWL_TOOL == OUR_CRAWLER:
            if type(item["actors"]) == list:
                item["actors"] = ",".join(item["actors"])
            if type(item["genres"]) == list:
                item["genres"] = ",".join(item["genres"])
            if type(item["writers"]) == list:
                item["writers"] = ",".join(item["writers"])
            self.movies.append(dict(item))
        else:
            if "Error" not in item["json"] and "Title" in item["json"]: # ignores error response
                self.movie_jsons.append(item["json"])

    def close_spider(self, spider):
        if CRAWL_TOOL == OUR_CRAWLER:
            with open(DATA_FILE_1, "w") as file:
                for movie in self.movies:
                    json.dump(movie, file)
                    file.write("\n")
                file.close()
        else:
            with open(DATA_FILE_2, "w") as file:
                for movie in self.movie_jsons:
                    json.dump(movie, file)
                    file.write("\n")
                file.close()
