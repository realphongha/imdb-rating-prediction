"""
Simple crawler to crawl movies' data from IMDB.
Phong Ha, October 2020
github.com/realphongha
"""

import scrapy


class MovieJson(scrapy.Item):
    # only json string, crawled item from OMDb API
    json = scrapy.Field()


class Movie(scrapy.Item):
    # item object for our crawler
    # prior data fields
    title = scrapy.Field()
    year = scrapy.Field()
    rated = scrapy.Field()
    runtime = scrapy.Field()
    genres = scrapy.Field()
    keywords = scrapy.Field()

    directors = scrapy.Field()
    writers = scrapy.Field()
    actors = scrapy.Field()
    producers = scrapy.Field()
    composers = scrapy.Field()
    cinematographers = scrapy.Field()
    film_editors = scrapy.Field()
    art_directors = scrapy.Field()

    # posterior data fields
    awards_oscar = scrapy.Field()
    awards_other = scrapy.Field()
    imdb_votes = scrapy.Field()
    metascore = scrapy.Field()
    imdb_rating = scrapy.Field()
