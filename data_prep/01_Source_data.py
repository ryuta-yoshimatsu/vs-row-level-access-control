# Databricks notebook source
# MAGIC %md # Vector Search demo: Prepare source Delta table
# MAGIC - Please notice you should not run this notebook in a serverless cluster

# COMMAND ----------

# MAGIC %pip install -U -qqqq langchain==0.2.11 tiktoken==0.7.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./00_Configs"

# COMMAND ----------

# MAGIC %md ### Part 1: Create table
# MAGIC
# MAGIC * Load a few Wikipedia articles.
# MAGIC * Chunk articles using a LangChain Text Splitter within a Spark UDF.
# MAGIC * Save to the source Delta table, with the Change Data Feed enabled.

# COMMAND ----------

# MAGIC %md #### Drop table if exists

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {source_table_fullname}")

# COMMAND ----------

# MAGIC %md #### Load articles

# COMMAND ----------

source_data_path = "dbfs:/databricks-datasets/wikipedia-datasets/data-001/en_wikipedia/articles-only-parquet"
source_df = spark.read.parquet(source_data_path).limit(10)
display(source_df)

# COMMAND ----------

# MAGIC %md #### Split articles into chunks and defining ACLs

# COMMAND ----------

from langchain.text_splitter import TokenTextSplitter
import pandas as pd
from pyspark.sql.functions import pandas_udf
from typing import Iterator
from pyspark.sql.types import IntegerType, StructField, StructType, StringType
import pyspark.sql.functions as F

def pandas_text_splitter(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
  """
  Apache Spark pandas UDF for chunking text in a scale-out pipeline.
  """
  token_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=20)
  for pdf in iterator:
    pdf["text"] = pdf["text"].apply(token_splitter.split_text)
    chunk_pdf = pdf.explode("text")
    chunk_pdf_with_index = chunk_pdf.reset_index().rename(columns={"index" : "chunk_id"})
    chunk_ids = chunk_pdf_with_index.groupby("chunk_id").cumcount()
    # Define ids with format "[ORIGINAL ID]_[CHUNK ID]"
    chunk_pdf_with_index["id"] = chunk_pdf_with_index["id"].astype("str") + "_" + chunk_ids.astype("str")
    yield chunk_pdf_with_index

  

# We need to adjust the "id" field to be a StringType instead of an IntegerType,
# because our IDs now have format "[ORIGINAL ID]_[CHUNK ID]".
schema_without_id = list(filter(lambda f: f.name != "id", source_df.schema.fields))
chunked_schema = StructType([
  StructField("id", StringType()),
  StructField("chunk_id", IntegerType())] + schema_without_id)  

chunked_df = (
  source_df
    .mapInPandas(pandas_text_splitter, schema=chunked_schema)
    .withColumnRenamed("text", "chunk")
    ################################ ACLs ######################################
    .withColumn("access_list", F.lit(["user_a", "user_b"]))## <--- Defining ACLS 
    ############################################################################
)

display(chunked_df)  

# COMMAND ----------

# MAGIC %md #### Save source Delta table

# COMMAND ----------

chunked_df.write.format("delta").mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(source_table_fullname)

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {source_table_fullname}"))

# COMMAND ----------

# MAGIC %md ## Part 2: Update table
# MAGIC
# MAGIC Load, chunk, and insert a few more articles into the source Delta table.
# MAGIC
# MAGIC This requires the previous cells of the notebook to have been run already.

# COMMAND ----------

source_df2 = spark.read.parquet(source_data_path).limit(20).exceptAll(source_df)
display(source_df2)

# COMMAND ----------

chunked_df2 = (
  source_df2
  .mapInPandas(pandas_text_splitter, schema=chunked_schema)
  .withColumnRenamed("text", "chunk")
  ################################ ACLs ################################################
  .withColumn("access_list", F.lit(["user_b", "user_c"]))## <--- Defining different ACLS 
  ######################################################################################
)

display(chunked_df2)

# COMMAND ----------

chunked_df2.write.format("delta").mode("append").saveAsTable(source_table_fullname)

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {source_table_fullname}"))