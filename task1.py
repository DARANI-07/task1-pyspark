pip install pyspark

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.ml.stat import Correlation

# Initialize Spark Session
spark = SparkSession.builder.appName("SimilaritySearch").getOrCreate()

# Load Dataset
df = spark.read.csv("BX-Users.csv", header=True, inferSchema=True)

# Text Preprocessing
tokenizer = Tokenizer(inputCol="text_column", outputCol="tokens")
df_tokens = tokenizer.transform(df)

stopword_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
df_clean = stopword_remover.transform(df_tokens)

hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features")
df_hashed = hashing_tf.transform(df_clean)

idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(df_hashed)
df_tfidf = idf_model.transform(df_hashed)

# Cosine Similarity
vectors = df_tfidf.select("tfidf_features").rdd.map(lambda row: row[0])
cosine_similarity = Correlation.corr(vectors, "pearson").head()[0]

print("Cosine Similarity Matrix:", cosine_similarity)
