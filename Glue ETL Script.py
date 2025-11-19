import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, to_date, upper, coalesce, lit
from awsglue.dynamicframe import DynamicFrame

## Initialize contexts
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# --- Define S3 Paths (Updated with your new names) ---
s3_input_path = "s3://handsonfinallanding-1/"
s3_processed_path = "s3://handsonfinalprocessed-1/processed-data/"
s3_analytics_path = "s3://handsonfinalprocessed-1/Athena Results/"

# --- Read the data from the S3 landing zone ---
dynamic_frame = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": [s3_input_path], "recurse": True},
    format="csv",
    format_options={"withHeader": True, "inferSchema": True},
)

# Convert to a standard Spark DataFrame for easier transformation
df = dynamic_frame.toDF()

# --- Perform Transformations ---
# 1. Cast 'rating' to integer and fill null values with 0
df_transformed = df.withColumn("rating", coalesce(col("rating").cast("integer"), lit(0)))

# 2. Convert 'review_date' string to a proper date type
df_transformed = df_transformed.withColumn("review_date", to_date(col("review_date"), "yyyy-MM-dd"))

# 3. Fill null review_text with a default string
df_transformed = df_transformed.withColumn("review_text",
    coalesce(col("review_text"), lit("No review text")))

# 4. Convert product_id to uppercase for consistency
df_transformed = df_transformed.withColumn("product_id_upper", upper(col("product_id")))


# --- Write the full transformed data to S3 (Good practice) ---
# This saves the clean, complete dataset to the 'processed-data' folder
glue_processed_frame = DynamicFrame.fromDF(df_transformed, glueContext, "transformed_df")
glueContext.write_dynamic_frame.from_options(
    frame=glue_processed_frame,
    connection_type="s3",
    connection_options={"path": s3_processed_path},
    format="csv"
)

# --- Run Spark SQL Query within the Job ---

# 1. Create a temporary view in Spark's memory
df_transformed.createOrReplaceTempView("product_reviews")

# 2. Run your SQL query: Average rating & review count per product (existing)
df_analytics_result = spark.sql("""
    SELECT 
        product_id_upper, 
        AVG(rating) as average_rating,
        COUNT(*) as review_count
    FROM product_reviews
    GROUP BY product_id_upper
    ORDER BY average_rating DESC
""")

# 3. Write the query's result DataFrame to your 'Athena Results' path
print(f"Writing analytics results to {s3_analytics_path}...")

# repartition(1) writes the result as a single file
analytics_result_frame = DynamicFrame.fromDF(df_analytics_result.repartition(1), glueContext, "analytics_df")
glueContext.write_dynamic_frame.from_options(
    frame=analytics_result_frame,
    connection_type="s3",
    connection_options={"path": s3_analytics_path + "avg_rating_per_product/"},
    format="csv"
)

# ------------------- ADDITIONAL REQUIRED QUERIES -------------------

# Query 2: Date-wise review count (total number of reviews per day)
df_date_wise = spark.sql("""
    SELECT
        review_date,
        COUNT(*) AS review_count
    FROM product_reviews
    GROUP BY review_date
    ORDER BY review_date
""")
print("Writing date-wise review count results...")
date_wise_frame = DynamicFrame.fromDF(df_date_wise.repartition(1), glueContext, "date_wise_df")
glueContext.write_dynamic_frame.from_options(
    frame=date_wise_frame,
    connection_type="s3",
    connection_options={"path": s3_analytics_path + "date_wise_review_count/"},
    format="csv"
)

# Query 3: Top 5 Most Active Customers (customers with most reviews)
df_top_customers = spark.sql("""
    SELECT
        customer_id,
        COUNT(*) AS reviews_count
    FROM product_reviews
    GROUP BY customer_id
    ORDER BY reviews_count DESC
    LIMIT 5
""")
print("Writing top 5 most active customers results...")
top_customers_frame = DynamicFrame.fromDF(df_top_customers.repartition(1), glueContext, "top_customers_df")
glueContext.write_dynamic_frame.from_options(
    frame=top_customers_frame,
    connection_type="s3",
    connection_options={"path": s3_analytics_path + "top_5_most_active_customers/"},
    format="csv"
)

# Query 4: Overall Rating Distribution (count for each star rating)
df_rating_dist = spark.sql("""
    SELECT
        rating,
        COUNT(*) AS rating_count
    FROM product_reviews
    GROUP BY rating
    ORDER BY rating
""")
print("Writing rating distribution results...")
rating_dist_frame = DynamicFrame.fromDF(df_rating_dist.repartition(1), glueContext, "rating_dist_df")
glueContext.write_dynamic_frame.from_options(
    frame=rating_dist_frame,
    connection_type="s3",
    connection_options={"path": s3_analytics_path + "rating_distribution/"},
    format="csv"
)

# End job
job.commit()