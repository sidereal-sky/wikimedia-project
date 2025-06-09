from pyspark.sql.functions import (
    col, length, coalesce, lit, when, year, month, 
    sum as spark_sum, avg, max as spark_max, count, 
    explode, split, regexp_replace
)
from pyspark.sql.types import StructType, StructField, StringType, LongType

# OPTIMIZED FOR 20-MINUTE RUNTIME
SAMPLE_SIZE = 250  # 250 titles = ~500 API calls = ~20 minutes

def enhanced_fetch_pageviews(title, start_date=START_DATE, end_date=END_DATE):
    url = f"{WIKI_API_BASE_URL}/{title}/monthly/{start_date}/{end_date}"
    headers = {'User-Agent': 'MyWikipediaScript/1.0 (test@gmail.com)'}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return (title, 0)
            
        data = response.json()
        
        if "items" in data and len(data["items"]) > 0:
            total_views = 0
            for item in data["items"]:
                if "views" in item and item["views"] is not None:
                    total_views += int(item["views"])
            views = total_views
        else:
            views = 0
            
        return (title, views)
        
    except Exception as e:
        return (title, 0)

def load_enhanced_pageviews(spark, titles_rdd):
    sample_titles = titles_rdd.take(SAMPLE_SIZE)
    print(f"Loading pageviews for {len(sample_titles)} titles...")
    
    enhanced_pageviews_list = []
    for i, title in enumerate(sample_titles):
        if i % 50 == 0:  # Progress update every 50 calls
            print(f"  Pageviews progress: {i}/{len(sample_titles)}")
        result = enhanced_fetch_pageviews(title)
        enhanced_pageviews_list.append(result)
    
    pageviews_schema = StructType([
        StructField("title", StringType(), True),
        StructField("extended_views", LongType(), True),
    ])
    
    enhanced_pageviews_df = spark.createDataFrame(enhanced_pageviews_list, schema=pageviews_schema)
    print(f"✓ Pageviews loaded: {enhanced_pageviews_df.count()} rows")
    return enhanced_pageviews_df

def enhanced_fetch_categories_with_count(title):
    url = "https://simple.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json", 
        "prop": "categories",
        "titles": title,
        "cllimit": "max",
    }
    headers = {'User-Agent': 'MyWikipediaScript/1.0 (test@gmail.com)'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        categories = []
        for page_id, page_data in pages.items():
            if page_id != "-1" and "categories" in page_data:
                categories = [cat["title"] for cat in page_data["categories"]]
        
        return (title, len(categories), str(categories[:5]))
    except Exception as e:
        return (title, 0, "[]")

def load_enhanced_categories(spark, titles_rdd):
    sample_titles = titles_rdd.take(SAMPLE_SIZE)
    print(f"Loading categories for {len(sample_titles)} titles...")
    
    enhanced_categories_list = []
    for i, title in enumerate(sample_titles):
        if i % 50 == 0:
            print(f"  Categories progress: {i}/{len(sample_titles)}")
        result = enhanced_fetch_categories_with_count(title)
        enhanced_categories_list.append(result)
    
    categories_schema = StructType([
        StructField("title", StringType(), True),
        StructField("category_count", LongType(), True),
        StructField("top_categories", StringType(), True),
    ])
    
    enhanced_categories_df = spark.createDataFrame(enhanced_categories_list, schema=categories_schema)
    print(f"✓ Categories loaded: {enhanced_categories_df.count()} rows")
    return enhanced_categories_df

def load_enhanced_edits_from_dump(spark, pages_df):
    print(f"Loading edit data from dump (no API calls)...")
    sample_pages = pages_df.limit(SAMPLE_SIZE)
    
    real_edits_data = sample_pages.select(
        col("title"),
        col("revision.id").alias("revision_id"),
        col("revision.parentid").alias("parent_revision_id"),
        col("revision.timestamp").alias("last_edit_timestamp"),
        col("revision.contributor.username").alias("contributor_username"),
        col("revision.contributor.id").alias("contributor_id"),
        col("revision.comment").alias("edit_comment"),
        length(coalesce(col("revision.text"), lit(""))).alias("content_length")
    )
    
    enhanced_edits_df = real_edits_data.select(
        col("title"),
        col("revision_id"),
        col("last_edit_timestamp"),
        col("contributor_username"),
        col("content_length"),
        when(col("revision_id").isNotNull(), 1).otherwise(0).alias("has_revisions"),
        when(col("contributor_username").isNotNull(), 1).otherwise(0).alias("has_contributors"),
        when(col("parent_revision_id").isNotNull(), 1).otherwise(0).alias("has_edit_history"),
        when(col("edit_comment").isNotNull() & (col("edit_comment") != ""), 1).otherwise(0).alias("has_edit_comments")
    )
    
    print(f"✓ Edit data loaded: {enhanced_edits_df.count()} rows")
    return enhanced_edits_df

# Load all datasets
print("🚀 Starting Wikipedia analysis...")
enhanced_pageviews_df = load_enhanced_pageviews(spark, titles_rdd)
enhanced_categories_df = load_enhanced_categories(spark, titles_rdd)
enhanced_edits_df = load_enhanced_edits_from_dump(spark, pages_df)

print("🔗 Joining datasets...")
combined_analysis = enhanced_pageviews_df.alias("views").join(
    enhanced_categories_df.alias("cats"),
    col("views.title") == col("cats.title"),
    "inner"
).join(
    enhanced_edits_df.alias("edits"),
    col("views.title") == col("edits.title"),
    "inner"
).select(
    col("views.title"),
    col("views.extended_views"),
    col("cats.category_count"),
    col("cats.top_categories"),
    col("edits.revision_id"),
    col("edits.last_edit_timestamp"),
    col("edits.contributor_username"),
    col("edits.content_length"),
    col("edits.has_revisions"),
    col("edits.has_contributors"),
    col("edits.has_edit_history"),
    col("edits.has_edit_comments")
)

print(f"✅ Analysis complete! Combined dataset: {combined_analysis.count()} rows")

# CHART 1: Top 20 Articles by Pageviews
top_pageviews = combined_analysis.orderBy(col("extended_views").desc()).limit(20)
display(top_pageviews)

# CHART 2: Pageviews Distribution
pageviews_distribution = combined_analysis.select(
    col("extended_views"),
    when(col("extended_views") == 0, "No Views")
    .when(col("extended_views") <= 100, "1-100 Views")
    .when(col("extended_views") <= 1000, "101-1K Views")
    .when(col("extended_views") <= 10000, "1K-10K Views")
    .when(col("extended_views") <= 100000, "10K-100K Views")
    .otherwise("100K+ Views").alias("view_range")
).groupBy("view_range").count().orderBy("count")
display(pageviews_distribution)

# CHART 3: Category Distribution
category_distribution = combined_analysis.select(
    when(col("category_count") == 0, "Uncategorized")
    .when(col("category_count") <= 5, "1-5 Categories")
    .when(col("category_count") <= 15, "6-15 Categories")
    .when(col("category_count") <= 30, "16-30 Categories")
    .otherwise("30+ Categories").alias("category_level")
).groupBy("category_level").count().orderBy("count")
display(category_distribution)

# CHART 4: Content Length vs Pageviews
content_vs_views = combined_analysis.select(
    col("title"),
    col("extended_views").alias("pageviews"),
    col("content_length"),
    col("contributor_username").alias("contributor")
).filter(col("pageviews") > 0)
display(content_vs_views)

# CHART 5: Edit Activity Analysis
edit_activity = combined_analysis.select(
    when(col("has_revisions") == 0, "No Revision Data")
    .when(col("has_contributors") == 0, "No Contributor Info") 
    .when(col("has_edit_comments") == 1, "Has Edit Comments")
    .when(col("has_edit_history") == 1, "Has Edit History")
    .otherwise("Basic Edit Info").alias("edit_info_level")
).groupBy("edit_info_level").count().orderBy("count")
display(edit_activity)

# CHART 6: Content Quality Score
content_quality = combined_analysis.select(
    col("title"),
    col("extended_views"),
    col("category_count"),
    col("content_length"),
    col("has_edit_history"),
    col("has_edit_comments"),
    (col("extended_views") * 0.001 + 
     col("category_count") * 2 + 
     col("content_length") * 0.01 +
     col("has_edit_history") * 50 +
     col("has_edit_comments") * 25).alias("quality_score")
).orderBy(col("quality_score").desc()).limit(25)
display(content_quality)

# CHART 7: Article Timeline Analysis
timeline_analysis = combined_analysis.select(
    col("title"),
    col("last_edit_timestamp"),
    year(col("last_edit_timestamp")).alias("edit_year"),
    month(col("last_edit_timestamp")).alias("edit_month"),
    col("extended_views"),
    col("content_length")
).filter(col("last_edit_timestamp").isNotNull())
display(timeline_analysis)

# CHART 8: Performance Metrics Summary
performance_summary = combined_analysis.agg(
    spark_sum("extended_views").alias("total_pageviews"),
    avg("extended_views").alias("avg_pageviews"),
    spark_max("extended_views").alias("max_pageviews"),
    spark_sum("category_count").alias("total_categories"),
    avg("category_count").alias("avg_categories"),
    avg("content_length").alias("avg_content_length"),
    spark_sum("has_contributors").alias("articles_with_contributors"),
    spark_sum("has_edit_history").alias("articles_with_edit_history"),
    count("*").alias("total_articles")
)
display(performance_summary)

# CHART 9: Top Categories Analysis
top_categories_expanded = combined_analysis.filter(col("category_count") > 0).select(
    col("title"),
    col("category_count"),
    col("extended_views"),
    explode(split(regexp_replace(col("top_categories"), "[\\[\\]']", ""), ", ")).alias("category")
).filter(col("category") != "").groupBy("category").agg(
    count("*").alias("article_count"),
    spark_sum("extended_views").alias("total_views"),
    avg("extended_views").alias("avg_views")
).orderBy(col("article_count").desc()).limit(20)
display(top_categories_expanded)

# CHART 10: Real Data Correlation Analysis
correlation_data = combined_analysis.select(
    col("extended_views").alias("pageviews"),
    col("category_count").alias("categories"), 
    col("content_length").alias("content_size"),
    col("has_edit_history").alias("has_history"),
    col("has_contributors").alias("has_contributor_info")
).filter(col("pageviews") > 0)
display(correlation_data)

# Save results
combined_analysis.write.mode("overwrite").option("header", "true").csv("dbfs:/FileStore/shared_uploads/andrei.ciobanu2508@stud.acs.upb.ro/enhanced_analysis_results")
