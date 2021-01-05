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
                item["actors"] = ",".join(map(lambda s: s.strip(), item["actors"]))
            if type(item["genres"]) == list:
                item["genres"] = ",".join(map(lambda s: s.strip(), item["genres"]))
            if type(item["writers"]) == list:
                item["writers"] = ",".join(map(lambda s: s.strip(), item["writers"]))
            if type(item["directors"]) == list:
                item["directors"] = ",".join(map(lambda s: s.strip(), item["directors"]))
            if type(item["producers"]) == list:
                item["producers"] = ",".join(map(lambda s: s.strip(), item["producers"]))
            if type(item["composers"]) == list:
                item["composers"] = ",".join(map(lambda s: s.strip(), item["composers"]))
            if type(item["cinematographers"]) == list:
                item["cinematographers"] = ",".join(map(lambda s: s.strip(), item["cinematographers"]))
            if type(item["film_editors"]) == list:
                item["film_editors"] = ",".join(map(lambda s: s.strip(), item["film_editors"]))
            if type(item["art_directors"]) == list:
                item["art_directors"] = ",".join(map(lambda s: s.strip(), item["art_directors"]))
            self.movies.append(dict(item))
        else:
            if "Error" not in item["json"] and "Title" in item["json"]:  # ignores error response
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
