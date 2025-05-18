from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, regexp_replace
from pyspark.sql.types import StringType
from transformers import pipeline
import os


def setup_spark_session():
    # Make sure we have a spark-temp directory
    os.makedirs("./spark-temp", exist_ok=True)
    
    return SparkSession.builder \
        .appName("Title Categorization") \
        .config("spark.driver.host", "localhost") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.local.dir", "./spark-temp") \
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

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
        
    try:
        spark = setup_spark_session()

        titles_df = spark.read.text(input_file).toDF("title")
        titles_df = titles_df.filter(col("title") != "")
        
        # Clean the titles to avoid parsing issues - replace double pipes with single pipes
        titles_df = titles_df.withColumn("title", regexp_replace(col("title"), "\\|\\|", "|"))

        # Create UDF for classification
        classify_udf = udf(lambda title: classify_title(title, candidate_labels), StringType())

        # Sample a small number of titles for testing to avoid memory issues
        sample_size = min(100, titles_df.count())
        titles_sample = titles_df.limit(sample_size)

        categorized_df = titles_sample.withColumn("category", classify_udf(col("title")))
        
        # Clean the category to avoid parsing issues with the delimiter
        categorized_df = categorized_df.withColumn("category", regexp_replace(col("category"), "\\|\\|", "|"))

        # Ensure output directory exists
        os.makedirs(output_file, exist_ok=True)
        
        # Create fallback simple CSV first
        simple_df = categorized_df.toPandas()
        simple_df.to_csv(f"{output_file}/categories.csv", sep=',', index=False)
        
        # Write results with proper escaping
        categorized_df.select("title", "category") \
            .write \
            .mode("overwrite") \
            .option("header", "true") \
            .option("delimiter", ",") \
            .option("quote", "\"") \
            .option("escape", "\\") \
            .csv(f"{output_file}/spark_output")
            
        print(f"Successfully categorized {sample_size} titles and saved to {output_file}")

        spark.stop()
        return True
    except Exception as e:
        print(f"Error during title categorization: {e}")
        # Create a fallback local CSV if Spark fails
        try:
            if os.path.exists(input_file):
                with open(input_file, 'r') as f:
                    titles = [line.strip() for line in f.readlines()[:20]]
                
                # Create a minimal output directory and CSV
                os.makedirs(output_file, exist_ok=True)
                with open(f"{output_file}/fallback.csv", 'w') as f:
                    f.write("title,category\n")  # Use comma instead of ||
                    for title in titles:
                        # Escape any commas in the title
                        safe_title = title.replace(",", "\\,").replace("\"", "\\\"")
                        f.write(f"\"{safe_title}\",uncategorized\n")
                print("Created fallback categorization file.")
        except Exception as inner_e:
            print(f"Error creating fallback file: {inner_e}")
        return False


def main():
    input_file = "titles.txt"
    output_file = "categorized_titles"

    categorize_titles_spark(input_file, output_file)


if __name__ == "__main__":
    main()