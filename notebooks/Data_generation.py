from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql.functions import col, avg, desc, count
import csv
import random

def generate_movie(movie_id):
    title = f"Movie {movie_id}"
    genre = random.choice(["Action", "Comedy", "Drama", "Thriller", "Sci-Fi"])
    return [movie_id, title, genre]

def generate_rating(user_id, movie_id):
    rating = random.randint(1, 5)
    return [user_id, movie_id, rating]

num_movies = 200
num_users = 100
max_ratings_per_user = 30

movies_data = [generate_movie(movie_id) for movie_id in range(1, num_movies + 1)]

ratings_data = []
for user_id in range(1, num_users + 1):
    num_ratings = random.randint(1, max_ratings_per_user)
    rated_movies = random.sample(range(1, num_movies + 1), num_ratings)
    for movie_id in rated_movies:
        ratings_data.append(generate_rating(user_id, movie_id))

movies_file_name = "custom_movies.csv"
ratings_file_name = "custom_ratings.csv"

with open(movies_file_name, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["movieId", "title", "genre"])
    writer.writerows(movies_data)

with open(ratings_file_name, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["userId", "movieId", "rating"])
    writer.writerows(ratings_data)

spark = SparkSession.builder \
    .appName("CombinedAnalysis") \
    .getOrCreate()

movies_schema = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("genre", StringType(), True)
])

ratings_schema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True)
])

movies_df = spark.read.csv(movies_file_name, header=True, schema=movies_schema)
ratings_df = spark.read.csv(ratings_file_name, header=True, schema=ratings_schema)

movies_df = movies_df.dropna(how='any')
ratings_df = ratings_df.dropna(how='any')

total_movies = movies_df.count()
total_ratings = ratings_df.count()

avg_ratings_per_movie = ratings_df.groupBy("movieId").agg(avg("rating").alias("avg_rating"))
top_rated_movies = avg_ratings_per_movie.join(movies_df, "movieId").orderBy(desc("avg_rating")).select("title", "avg_rating").limit(10)

print("Basic Analysis:")
print("Total number of movies:", total_movies)
print("Total number of ratings:", total_ratings)
print("\nAverage rating for each movie:")
avg_ratings_per_movie.show()
print("\nTop-rated movies:")
top_rated_movies.show()

from pyspark.ml.recommendation import ALS

als = ALS(maxIter=10, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(ratings_df)

user_id = 1
num_recommendations = 5
user_recommendations = model.recommendForAllUsers(num_recommendations)
user_recommendations = user_recommendations.filter(col("userId") == user_id)
recommended_movie_ids = [row.movieId for row in user_recommendations.collect()[0].recommendations]

print("\nRecommendation Analysis:")
print(f"Top {num_recommendations} movie recommendations for user {user_id}:")
print(recommended_movie_ids)

genre_avg_ratings = ratings_df.join(movies_df, "movieId") \
    .groupBy("genre") \
    .agg(avg("rating").alias("avg_rating"))

genre_rating_counts = ratings_df.join(movies_df, "movieId") \
    .groupBy("genre") \
    .agg(count("rating").alias("rating_count")) \
    .orderBy(desc("rating_count"))

user_movie_counts = ratings_df.groupBy("userId") \
    .agg(count("movieId").alias("movie_count")) \
    .orderBy(desc("movie_count"))

print("\nAdditional Analysis:")
print("Average rating for each genre:")
genre_avg_ratings.show(truncate=False)
print("Most popular genres based on the number of ratings:")
genre_rating_counts.show(truncate=False)
print("Users who have rated the most movies:")
user_movie_counts.show(truncate=False)

spark.stop()
