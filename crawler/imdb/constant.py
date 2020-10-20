"""
Simple crawler to crawl movies' data from IMDB.
Phong Ha, October 2020
github.com/realphongha
"""

# urls:
IMDB = "https://www.imdb.com"
URL_TEMPLATE = "http://www.omdbapi.com/?i=%s&apikey=%s"

# advanced search link: https://www.imdb.com/search/title/#details?ref_=kw_brw_2
# search links:
US = "https://www.imdb.com/search/title/?countries=us" # movies from the US
US_MOVIES = "https://www.imdb.com/search/title/?title_type=feature&countries=us" # only feature films from the US
IMDB_TOP100 = "https://www.imdb.com/search/title/?groups=top_100"
# crawl limit:
MAX_MOVIES_TO_CRAWL = 15000
# select tool to crawl (our own awesome crawler or via OMDb API)
OUR_CRAWLER = 1
OMDB_API = 2
CRAWL_TOOL = OUR_CRAWLER # 1 means our own crawler, 2 means using OMDb API
assert CRAWL_TOOL == OUR_CRAWLER or CRAWL_TOOL == OMDB_API, "Invalid crawling tool!"
# storage:
DATA_FILE_1 = "data1.json" # for our own crawler
DATA_FILE_2 = "data2.json" # for data get from OMDb API