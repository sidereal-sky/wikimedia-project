import os
import sys
import glob

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType

from insights import *

DUMPS_BASE_DIR = "./chunks"
DUMPS_FILES = "./chunks/*.xml.bz2"


def main():
    if not os.path.exists(DUMPS_BASE_DIR):
        print("Error: The 'chunks' folder does not exist. Please create it and add XML dump files.")
        sys.exit(1)

    spark = SparkSession.builder \
        .appName("Wikipedia dump parser") \
        .config("spark.jars.packages", "com.databricks:spark-xml_2.12:0.18.0") \
        .config("spark.hadoop.security.authorization", "false") \
        .config("spark.hadoop.security.authentication", "simple") \
        .config("spark.driver.host", "localhost") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.local.dir", "./spark-temp") \
        .getOrCreate()

    page_schema = StructType([
        StructField("id", LongType(), True),
        StructField("title", StringType(), True),
        StructField("revision", StructType([
            StructField("id", LongType(), True),
            StructField("parentid", LongType(), True),
            StructField("timestamp", StringType(), True),
            StructField("comment", StringType(), True),
            StructField("contributor", StructType([
                StructField("username", StringType(), True),
                StructField("id", LongType(), True),
            ]), True),
            StructField("model", StringType(), True),
            StructField("format", StringType(), True),
            StructField("text", StringType(), True),
            StructField("sha1", StringType(), True),
        ]), True),
    ])

    # Create Spark temp directory if it doesn't exist
    os.makedirs("./spark-temp", exist_ok=True)
    
    titles = []
    titles_df = None

    try:
        # Check if there are any XML files in the chunks directory
        xml_files = glob.glob(DUMPS_FILES)
        
        if xml_files:
            print(f"Found {len(xml_files)} XML dump files.")
            pages_df = spark.read \
                .format("xml") \
                .option("rootTag", "pages") \
                .option("rowTag", "page") \
                .load(DUMPS_FILES, schema=page_schema)

            pages_df.printSchema()

            pages = pages_df.select(
                pages_df.id.cast("long").alias("id"),
                pages_df.title.alias("title"),
                pages_df.revision.timestamp.alias("timestamp"),
            )

            pages.show(10)

            titles = pages.select("title").rdd.flatMap(lambda row: [row.title]).collect()
            
            # Save titles to file
            with open("titles.txt", "w") as f:
                for title in titles:
                    f.write(title + "\n")
        else:
            print("Warning: No XML files found in the chunks directory.")
            if os.path.exists("titles.txt"):
                with open("titles.txt", "r") as f:
                    titles = [line.strip() for line in f.readlines()]
                print(f"Loaded {len(titles)} titles from titles.txt")
            else:
                print("Error: No titles.txt file found either. Please add XML files or create a titles.txt file.")
            
    except Exception as e:
        print(f"Error processing XML files: {e}")
        print("Continuing with existing titles.txt if available...")
        if os.path.exists("titles.txt"):
            with open("titles.txt", "r") as f:
                titles = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(titles)} titles from titles.txt")

    # Create a DataFrame and RDD from titles for analytics functions
    if titles and len(titles) > 0:
        # Create DataFrame and RDD from the titles list for analytics
        titles_df = spark.createDataFrame([(title,) for title in titles], ["title"])
        titles_rdd = titles_df.select("title").rdd.flatMap(lambda row: [row.title])
        
        # Uncomment to view pageviews per article
        print("Fetching pageviews...")
        load_pageviews(spark, titles_rdd)

        # Uncomment to view categories per article
        print("Fetching categories...")
        load_categories(spark, titles_rdd)

        # Uncomment to view edits per article - might take a bit longer
        print("Fetching edits...")
        load_edits(spark, titles_rdd)

        # Uncomment to view global trends
        print("Fetching global trends...")
        load_global_trends(spark, titles_rdd)

    spark.stop()


if __name__ == "__main__":
    main()
