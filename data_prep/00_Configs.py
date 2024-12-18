# Databricks notebook source
# MAGIC %pip install -U -qqqq mlflow==2.15.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow

# Load model config
config = mlflow.models.ModelConfig(development_config="../config.yaml")
# Source Delta table
catalog = config.get("catalog")
schema = config.get("schema")
source_table = config.get("source_table")
source_table_fullname = f"{catalog}.{schema}.{source_table}"

print("Source Delta table:")
print(f"  source_table: {source_table}")
print(f"  source_table_fullname: {source_table_fullname}")

# COMMAND ----------

vs_endpoint = config.get("vs_endpoint")
vs_index = config.get("index_name")
vs_index_fullname = f"{catalog}.{schema}.{vs_index}"

print(f"Vector index:")
print(f"  vs_catalog: {catalog}")
print(f"  vs_schema: {schema}")
print(f"  vs_index: {vs_index}")
print(f"  vs_index_fullname: {vs_index_fullname}")
print(f"  vs_endpoint: {vs_endpoint}")

# COMMAND ----------

