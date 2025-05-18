import requests
from pyspark.sql.types import StructType, StructField, LongType, StringType

START_DATE = "20241201"
END_DATE = "20241231"

WIKI_API_BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/simple.wikipedia/all-access/all-agents"


def fetch_pageviews(title, start_date=START_DATE, end_date=END_DATE):
    """
    Fetch pageview data for a given page title using Wikimedia Pageviews API.
    """

    url = f"{WIKI_API_BASE_URL}/{title}/monthly/{start_date}/{end_date}"
    headers = {
        'User-Agent': 'MyWikipediaScript/1.0 (test@gmail.com)'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        views = sum(item["views"] for item in data["items"])
        return (title, views)
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data for {title}: {e}")
        return (title, None)


def load_pageviews(spark, titles_rdd):
    """
    Display pageviews for a given page title.
    """
    pageviews_rdd = titles_rdd.map(lambda title: fetch_pageviews(title)).take(100)

    pageviews_schema = StructType([
        StructField("title", StringType(), True),
        StructField("views", LongType(), True),
    ])
    pageviews_df = spark.createDataFrame(pageviews_rdd, schema=pageviews_schema)

    pageviews_df.show(100)


def fetch_categories(title):
    """
    Fetch categories for a given page title using Wikimedia Categories API.
    """
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "categories",
        "titles": title,
        "cllimit": "max",
    }
    headers = {
        'User-Agent': 'MyWikipediaScript/1.0 (test@gmail.com)'
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        categories = []
        for page_id, page_data in pages.items():
            if "categories" in page_data:
                categories = [cat["title"] for cat in page_data["categories"]]
        return (title, categories)
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch categories for {title}: {e}")
        return (title, None)


def load_categories(spark, titles_rdd):
    """
    Display categories for a given page title.
    """
    categories_rdd = titles_rdd.map(lambda title: fetch_categories(title)).take(100)

    categories_schema = StructType([
        StructField("title", StringType(), True),
        StructField("categories", StringType(), True),
    ])

    categories_rdd = [(title, str(categories)) for title, categories in categories_rdd]
    categories_df = spark.createDataFrame(categories_rdd, schema=categories_schema)
    categories_df.show(100, truncate=False)


def fetch_global_trends(project="en.wikipedia", access="all-access", year="2024", month="12", day="01"):
    """
    Fetch monthly global top articles using Wikimedia API.
    """
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{project}/{access}/{year}/{month}/{day}"
    headers = {
        'User-Agent': 'MyWikipediaScript/1.0 (test@gmail.com)'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Extract article titles and view counts
        articles = [
            (article["article"], article["views"]) for article in data["items"][0]["articles"]
        ]
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch global trends: {e}")
        return None


def filter_trending_articles(trending_articles, dump_titles):
    """
    Filters trending articles to only include those present in the Wikipedia dumps.
    """
    dump_titles_set = set(dump_titles)
    filtered_articles = [
        (title, views) for title, views in trending_articles if title in dump_titles_set
    ]
    return filtered_articles


def load_global_trends(spark, titles_rdd):
    """
    Display global trending articles from the Wikipedia dumps.
    """
    trending_articles = fetch_global_trends()
    filtered_trending_articles = filter_trending_articles(trending_articles, titles_rdd.collect())

    trending_articles_schema = StructType([
        StructField("title", StringType(), True),
        StructField("views", LongType(), True),
    ])

    trending_articles_df = spark.createDataFrame(filtered_trending_articles, schema=trending_articles_schema)
    trending_articles_df.show(100)


def fetch_edits(title):
    """
    Fetch edit data for a given page title using Wikimedia Edits API.
    """
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "rvlimit": "max",
    }
    headers = {
        'User-Agent': 'MyWikipediaScript/1.0 (testgmail.com)'
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        page_id = list(data['query']['pages'].keys())[0]
        revisions = data['query']['pages'][page_id].get('revisions', [])

        # Return the number of revisions (edits)
        return (title, len(revisions))
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch edits for {title}: {e}")
        return (title, None)


def load_edits(spark, titles_rdd):
    """
    Display the number of edits for a given page title.
    """
    edits_rdd = titles_rdd.map(lambda title: fetch_edits(title)).take(100)

    edits_schema = StructType([
        StructField("title", StringType(), True),
        StructField("edits", LongType(), True),
    ])
    edits_df = spark.createDataFrame(edits_rdd, schema=edits_schema)

    edits_df.show(100)
