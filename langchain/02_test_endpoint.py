# Databricks notebook source
# MAGIC %md
# MAGIC # Test Endpoint
# MAGIC This notebooks calls the endpoint deployed in the ``01_define_ai_pipeline_endpoint`` passing different ``users`` to test if the ACLs are applied
# MAGIC
# MAGIC <b>NOTE</b>: the deployed endpoint should be up and runing before this notebook is executed

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow==2.15.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Test deployed EndPoint

# COMMAND ----------

# MAGIC %md
# MAGIC Helper function to call endpoint passing different security groups

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json
import mlflow


# Loading config
config_path = os.path.join(os.getcwd(), "../../config.yaml")
config = mlflow.models.ModelConfig(development_config=config_path)

## Defining endpoint name
serving_endpoint_name = config.get("endpoint_name")

## Getting the workspace host name
host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"


def call_endpoint(query: str, user: str):

    url = f"{host}/serving-endpoints/{serving_endpoint_name}/invocations"

    headers = {
        "Authorization": f'Bearer {dbutils.secrets.get("udhay_secrets_scope", "pat")}',
        "Content-Type": "application/json",
    }

    vs_filters = {
        "access_list": [user],
    }

    model_input_sample = {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ],
        "filters": vs_filters,
    }

    data_json = json.dumps(
       model_input_sample,
    )
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)

    if response.status_code != 200:
        return f"{response.text}"
    return json.loads(response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ###When passing ``user_a`` has no permission to the excerpt mentioning the 'Hymn to Zeus' (another excerpt will be returned) 

# COMMAND ----------

call_endpoint(query="what is the Hymn to Zeus?", user="user_a")

# COMMAND ----------

# MAGIC %md
# MAGIC ###When passing ``user_c`` has permission to the excerpt mentioning the 'Hymn to Zeus'

# COMMAND ----------

call_endpoint(query="what is the Hymn to Zeus?", user="user_c")

# COMMAND ----------

# MAGIC %md
# MAGIC ###When passing ``user_b`` has permission to all excerpts

# COMMAND ----------

call_endpoint(query="what is the Hymn to Zeus?", user="user_b")