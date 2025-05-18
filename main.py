from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType
import os
import sys

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

    titles_rdd = pages.select("title").rdd.flatMap(lambda row: [row.title])

    # # Save the titles RDD to a text file, one title per line
    # titles_rdd.saveAsTextFile("titles.txt")

    spark.stop()


if __name__ == "__main__":
    main()
