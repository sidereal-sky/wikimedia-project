from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from transformers import pipeline


def setup_spark_session():
    return SparkSession.builder \
        .appName("Title Categorization") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()


def setup_zero_shot_classifier():
    """
    Setup zero-shot classification pipeline.
    Broadcast the pipeline to worker nodes to improve performance.
    """
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def classify_title(title, candidate_labels):
    try:
        # Import pipeline here to avoid serialization issues
        from transformers import pipeline

        # Recreate the classifier for each worker
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        result = classifier(title, candidate_labels)

        return f"{result['labels'][0]} ({result['scores'][0]:.2f})"

    except Exception as e:
        print(f"Error processing title '{title}': {e}")
        return "uncategorized"


def categorize_titles_spark(input_file, output_file):
    candidate_labels = [
        "technology", "science", "history", "art", "sports", "politics", "education",
        "medicine", "business", "culture", "literature", "health", "finance", "law",
        "music", "engineering", "mathematics", "environment", "philosophy", "economics",
        "media", "psychology", "geography", "agriculture", "space", "sociology"
    ]

    spark = setup_spark_session()

    titles_df = spark.read.text(input_file).toDF("title")
    titles_df = titles_df.filter(col("title") != "")

    # Create UDF for classification
    classify_udf = udf(lambda title: classify_title(title, candidate_labels), StringType())

    categorized_df = titles_df.withColumn("category", classify_udf(col("title")))

    categorized_df.select("title", "category") \
        .write \
        .mode("overwrite") \
        .option("header", "true") \
        .option("delimiter", "||") \
        .csv(output_file)

    spark.stop()


def main():
    input_file = "titles.txt"
    output_file = "categorized_titles"

    categorize_titles_spark(input_file, output_file)


if __name__ == "__main__":
    main()