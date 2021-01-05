"""
Simple crawler to crawl movies' data from IMDB.
Phong Ha, October 2020
github.com/realphongha

We use OMDb API to get data more easily: http://www.omdbapi.com/
You need OMDb API key to start crawling.
"""

import scrapy
import json
import logging
from ..items import *
from ..constant import *
from scrapy.exceptions import CloseSpider

# turns off debug log:
# logging.getLogger('scrapy').setLevel(logging.WARNING)
# logging.getLogger('scrapy').propagate = False # turns off logging


class ImdbSpider(scrapy.Spider):

    handle_httpstatus_list = [401]
    name = "imdb"

    def make_url(self, imdb_id):
        # makes url request from movie id and OMDb API secret key
        return URL_TEMPLATE % (imdb_id, self.OMDb_API_key)

    def stop_crawling(self):
        # STOP!!!
        raise CloseSpider("Done!")

    def start_requests(self):
        if CRAWL_TOOL == OUR_CRAWLER:
            yield scrapy.Request(url=US_MOVIES, callback=self.parse_search_page)
        else: # CRAWL_TOOL == OMDB_API
            self.OMDb_API_key = input("Enter your OMDb API key to start crawling: ").strip()
            yield scrapy.Request(url=US_MOVIES, callback=self.parse_search_page_OMDb)

    # our own crawler:

    def parse_search_page(self, response):
        # parses the search page to generate links to movie pages
        movies = response.xpath("//*[@id='main']/div[@class='article']/div/div/div/div/h3/a/@href").getall()
        for movie in movies:
            yield scrapy.Request(url=IMDB + str(movie), callback=self.parse_movie_page)

        next_page = response.xpath("//*[@id='main']/div[@class='article']/div[@class='desc']/a[@class='lister-page-next next-page']/@href").get()
        print("NEXT PAGE:", next_page)
        yield scrapy.Request(url=IMDB + str(next_page), callback=self.parse_search_page)

    def parse_movie_page(self, response):
        # parses the movie page to get data
        json_response = json.loads(response.css("script[type='application/ld+json']::text").get())
        year = response.xpath("//*[@id='titleYear']/a/text()").get()
        if type(json_response["director"]) == list:
            directors = []
            for director in json_response["director"]:
                directors.append(director["name"])
        elif type(json_response["director"]) == dict:
            directors = json_response["director"]["name"]
        if type(json_response["creator"]) == list:
            writers = []
            for writer in json_response["creator"]:
                if writer["@type"] == "Person":
                    writers.append(writer["name"])
        elif type(json_response["creator"]) == dict:
            writers = json_response["creator"]["name"]
        if type(json_response["actor"]) == list:
            actors = []
            for actor in json_response["actor"]:
                actors.append(actor["name"])
        elif type(json_response["actor"]) == dict:
            actors = json_response["actor"]["name"]
        metascore = response.css("div.titleReviewBarItem a div span::text").get()
        try:
            awards_oscar = " ".join(response.xpath("//*[@id='titleAwardsRanks']/span[1]/b/text()").get().split())
        except AttributeError:
            awards_oscar = None
        try:
            awards_other = " ".join(response.xpath("//*[@id='titleAwardsRanks']/span[2]/text()").get().split())
            if awards_other == "":
                awards_other = None
        except AttributeError:
            awards_other = None
        if "duration" in json_response:
            duration = json_response["duration"]
        else:
            duration = None
        if "aggregateRating" not in json_response:
            return
        yield Movie(title=json_response["name"], year=year, rated=json_response["contentRating"],
                    runtime=json_response["duration"], genres=json_response["genre"], directors=directors,
                    writers=writers, actors=actors, awards_oscar=awards_oscar, awards_other=awards_other,
                    imdb_votes=json_response["aggregateRating"]["ratingCount"], metascore=metascore,
                    keywords=json_response["keywords"], imdb_rating=json_response["aggregateRating"]["ratingValue"])
        # custom crawler using selectors but inconsistent:

        # title = response.xpath("//*[@id='title-overview-widget']/div/div/div/div/div[@class='title_wrapper']/h1/text()").get().strip()
        # year = response.xpath("//*[@id='titleYear']/a/text()").get()
        # rated = response.xpath("//*[@id='title-overview-widget']/div/div/div/div/div[@class='title_wrapper']/div[@class='subtext']/text()").get(1).strip()
        # runtime = response.xpath("//*[@id='titleDetails']/div[@class='txt-block']/time/text()").get()
        # genres = ",".join(response.xpath("//*[@id='titleStoryLine']/div[4]/a/text()").getall())
        # #
        # directors_raw = response.xpath(
        #     "//*[@id='title-overview-widget']/div[@class='plot_summary_wrapper']/div[@class='plot_summary ']/div[2]/a/text()").getall()
        # directors = []
        # for director in directors_raw:
        #     if "more credit" not in director:
        #         directors.append(director)
        # directors = ", ".join(directors)
        # writers_raw = response.xpath("//*[@id='title-overview-widget']/div[@class='plot_summary_wrapper']/div[@class='plot_summary ']/div[3]/a/text()").getall()
        # writers = []
        # for writer in writers_raw:
        #     if "more credit" not in writer:
        #         writers.append(writer)
        # writers = ", ".join(writers)
        # #
        # actors_raw = response.xpath(
        #     "//*[@id='title-overview-widget']/div[@class='plot_summary_wrapper']/div[@class='plot_summary ']/div[4]/a/text()").getall()
        # actors = []
        # for actor in actors_raw:
        #     if "See full cast" not in actor:
        #         actors.append(actor)
        # actors = ", ".join(actors)
        # try:
        #     awards_oscar = " ".join(response.xpath("//*[@id='titleAwardsRanks']/span[1]/b/text()").get().split())
        # except AttributeError:
        #     awards_oscar = None
        # try:
        #     awards_other = " ".join(response.xpath("//*[@id='titleAwardsRanks']/span[2]/text()").get().split())
        # except AttributeError:
        #     awards_other = None
        # imdb_votes = response.xpath("//*[@id='title-overview-widget']/div/div/div/div/div[@class='imdbRating']/a/span/text()").get()
        # metascore = response.css("div.titleReviewBarItem a div span::text").get()
        # imdb_rating = response.css("div.ratingValue strong span::text").get()
        # yield Movie(title=title, year=year, rated=rated, runtime=runtime, genres=genres, directors=directors,
        #             writers=writers, actors=actors, awards_oscar=awards_oscar, awards_other=awards_other,
        #             imdb_votes=imdb_votes, metascore=metascore, imdb_rating=imdb_rating)

    # handles via OMDb API:
    def parse_search_page_OMDb(self, response):
        # parses the search page to generate links to movies' json data on OMDb
        movie_ids = response.xpath("//*[@id='main']/div[@class='article']/div/div/div/div/div[@class='ribbonize']/@data-tconst").getall()
        for movie_id in map(str, movie_ids):
            yield scrapy.Request(url=self.make_url(movie_id), callback=self.parse_json_OMDb)

        next_page = response.xpath("//*[@id='main']/div[@class='article']/div[@class='desc']/a[@class='lister-page-next next-page']/@href").get()
        print("NEXT", next_page)
        yield scrapy.Request(url=IMDB + str(next_page), callback=self.parse_search_page_OMDb)

    def parse_json_OMDb(self, response):
        # parses movie data from OMDb API:
        json_response = json.loads(response.body)
        if json_response["imdbRating"] == "N/A":
            return
        yield MovieJson(json=json_response)
