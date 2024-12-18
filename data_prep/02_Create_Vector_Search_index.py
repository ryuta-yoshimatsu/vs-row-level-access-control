# Databricks notebook source
# MAGIC %md # Vector Search demo: Create and manage vector index
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall -qqqq databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Loading configurations

# COMMAND ----------

# MAGIC %run "./00_Configs"

# COMMAND ----------

# MAGIC %md
# MAGIC ###Instantiating VS Client

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# COMMAND ----------

# MAGIC %md ## Create vector index

# COMMAND ----------

import mlflow

config = mlflow.models.ModelConfig(development_config="../config.yaml")

index = vsc.create_delta_sync_index(
  endpoint_name=vs_endpoint,
  source_table_name=source_table_fullname,
  index_name=vs_index_fullname,
  pipeline_type='TRIGGERED',
  primary_key="id",
  embedding_source_column="chunk",
  embedding_model_endpoint_name="databricks-bge-large-en"
)
index.describe()