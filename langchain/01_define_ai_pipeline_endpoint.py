# Databricks notebook source
# MAGIC %md
# MAGIC ##Simple implementation of Vector Search record level ACLs with langchain
# MAGIC This notebook defines, registers, an deploys a langchain model, leveraging LCEL(Langchain Expression Language) and Vector Search filtering to implement record level ACLs:
# MAGIC - The chain only returns records when the ``user`` is present in the ``access_list`` column
# MAGIC - The client app passes the ``user`` in the REST API call as a ``parameter``
# MAGIC - For simplicity sake, no generation LLM is used.  The results from Vector Search are directly returned

# COMMAND ----------

# MAGIC %md
# MAGIC ###Install dependencies

# COMMAND ----------

# MAGIC %pip install -U -qqqq --force-reinstall databricks-sdk==0.29.0 databricks-vectorsearch==0.40 mlflow[databricks]==2.15.0 langchain==0.2.16 langchain_community==0.2.16
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Defining chain
# MAGIC
# MAGIC When the chain is invoked by the REST API call:
# MAGIC - The ``user`` is obtained from the ``params`` received in the request and passed down into the chain via config
# MAGIC - The Vector Search query uses a filter to avoid returning any records not having the requesting ``user`` in the ``access_list`` column 

# COMMAND ----------

# MAGIC %%writefile chain.py
# MAGIC import os
# MAGIC from typing import Dict, List, Optional
# MAGIC import mlflow
# MAGIC from operator import itemgetter
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from langchain.schema.runnable import RunnableLambda
# MAGIC from langchain_community.chat_models import ChatDatabricks
# MAGIC from langchain_community.vectorstores import DatabricksVectorSearch
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.prompts import PromptTemplate
# MAGIC from langchain_core.runnables import RunnablePassthrough, RunnableSequence
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain.schema.runnable import RunnableLambda
# MAGIC from langchain_core.runnables import ConfigurableField
# MAGIC from langchain_core.documents.base import Document
# MAGIC
# MAGIC config_path = os.path.join(os.getcwd(), "../../config.yaml")
# MAGIC config = mlflow.models.ModelConfig(development_config=config_path)
# MAGIC
# MAGIC
# MAGIC def extract_user_query_string(input: Dict) -> str:
# MAGIC     """
# MAGIC     Extracts the user's query string from the input data.
# MAGIC
# MAGIC     Args:
# MAGIC         input: The input data containing user messages.
# MAGIC
# MAGIC     Returns:
# MAGIC         The content of the last user message.
# MAGIC     """
# MAGIC     return input["messages"][-1]["content"]
# MAGIC
# MAGIC
# MAGIC def create_configurable_with_filters(input: Dict) -> Dict:
# MAGIC     """
# MAGIC     create configurable object with filters.
# MAGIC
# MAGIC     Args:
# MAGIC         input: The input data containing filters.
# MAGIC
# MAGIC     Returns:
# MAGIC         A configurable object with filters added to the search_kwargs.
# MAGIC     """
# MAGIC     configurable = {
# MAGIC         "configurable": {
# MAGIC             "search_kwargs": {
# MAGIC                 "k": 2,
# MAGIC                 "filters": input["filters"],
# MAGIC             }
# MAGIC         }
# MAGIC     }
# MAGIC     return configurable
# MAGIC
# MAGIC
# MAGIC def extract_pagecontent(docs: List[Document]) -> List[str]:
# MAGIC     """
# MAGIC     Extracts page content from a list of langchain documents.
# MAGIC
# MAGIC     Args:
# MAGIC         docs: A list of documents.
# MAGIC
# MAGIC     Returns:
# MAGIC         A list of page content extracted from the documents.
# MAGIC     """
# MAGIC     return [doc.page_content for doc in docs]
# MAGIC
# MAGIC
# MAGIC ############
# MAGIC # Connect to the Vector Search Index
# MAGIC ############
# MAGIC vs_client = VectorSearchClient(disable_notice=True)
# MAGIC vs_index = vs_client.get_index(
# MAGIC     endpoint_name=config.get("vs_endpoint"), index_name=f'{config.get("catalog")}.{config.get("schema")}.{config.get("index_name")}'
# MAGIC )
# MAGIC
# MAGIC
# MAGIC ############
# MAGIC # Turn the Vector Search index into a LangChain retriever
# MAGIC ############
# MAGIC vector_search_as_retriever = DatabricksVectorSearch(
# MAGIC     vs_index,
# MAGIC     text_column="chunk",
# MAGIC     columns=[
# MAGIC         "chunk",
# MAGIC     ],
# MAGIC ).as_retriever()
# MAGIC
# MAGIC ############
# MAGIC # Convert the retriever into configurable retriever so that run time parameters such as filters 
# MAGIC # can be  passed down during chain invoke
# MAGIC ############
# MAGIC configurable_vs_retriever = vector_search_as_retriever.configurable_fields(
# MAGIC     search_kwargs=ConfigurableField(
# MAGIC         id="search_kwargs",
# MAGIC         name="Search Kwargs",
# MAGIC         description="The search kwargs to use",
# MAGIC     )
# MAGIC )
# MAGIC
# MAGIC
# MAGIC ############
# MAGIC # RAG Chain
# MAGIC ############
# MAGIC chain_with_configurable = (
# MAGIC     {"messages": itemgetter("messages"), "filters": itemgetter("filters")}
# MAGIC     | RunnableLambda(
# MAGIC         lambda input: configurable_vs_retriever.invoke(
# MAGIC             extract_user_query_string(input), config=create_configurable_with_filters(input)
# MAGIC         )
# MAGIC     )
# MAGIC     | extract_pagecontent
# MAGIC )
# MAGIC
# MAGIC ## Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=chain_with_configurable)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Log AI Pipeline
# MAGIC Log ``chain`` in ``mlflow``

# COMMAND ----------

# infer mlflow signature 
import mlflow
from mlflow.models import infer_signature
from langchain.docstore.document import Document

vs_filters = {
    "access_list": ["user_b"],
}


model_input_sample = {
    "messages": [
        {
            "role": "user",
            "content": "Greek myths?",
        }
    ],
    "filters": vs_filters,
}

output_sample = ["output text chunk1"]

signature = infer_signature(model_input_sample, output_sample)

# COMMAND ----------

import mlflow
import os

config_path = "/Workspace/Users/user-email/vs-row-level-access-control/config.yaml"

# Log the model to MLflow
with mlflow.start_run():
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(os.getcwd(), 'chain.py'),  # Chain code file e.g., /path/to/the/chain.py 
        model_config=config_path,  # Chain configuration 
        artifact_path="chain",  # Required by MLflow
        input_example=model_input_sample,  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        example_no_conversion=True, # Required by MLflow to use the input_example as the chain's schema
        signature=signature
    )
    print(f"MLflow Run: {logged_chain_info.run_id}")
    print(f"Model URI: {logged_chain_info.model_uri}")

# COMMAND ----------

# Test the chain locally
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
result = chain.invoke(model_input_sample)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Register AI Pipeline in UC
# MAGIC
# MAGIC Register ``chain`` in Unity Catalog (using the ``catalog``, ``schema``, and ``model_name`` specified in the ``config.yaml`` file)

# COMMAND ----------

config = mlflow.models.ModelConfig(development_config=config_path)
uc_model_name = f"{config.get('catalog')}.{config.get('schema')}.{config.get('model_name')}"

# Register the model to the Unity Catalog
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, 
                                                 name=uc_model_name, )

# COMMAND ----------

# MAGIC %md
# MAGIC ###Test registered AI Pipeline
# MAGIC
# MAGIC Loading the registered model and running a quick test, before the model is deployed to an endpoint

# COMMAND ----------

### Test the registered model
model = mlflow.pyfunc.load_model(f"models:/{uc_model_name}/{uc_registered_model_info.version}")


model.predict(model_input_sample)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Deploy endpoint
# MAGIC
# MAGIC Deploying registered model to an endpoint (based on the ``endpoint_name`` defined in the ``config.yaml`` file)

# COMMAND ----------

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize, ServedModelInputWorkloadType

## Defining endpoint name
serving_endpoint_name = config.get("endpoint_name")

## Getting latest version of the model
latest_model_version = uc_registered_model_info.version

## Getting the workspace host name
host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"

w = WorkspaceClient()

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

## Deploying
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=uc_model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            workload_type=ServedModelInputWorkloadType.CPU,
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_HOST":  host,
                "DATABRICKS_TOKEN": token, 
            }
        )
    ]
)

# Checking if model exists. If so, update it, otherwise create it
existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create(name=serving_endpoint_name, config=endpoint_config) 
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config(served_models=endpoint_config.served_models, name=serving_endpoint_name) 
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC ###Please note you should wait until the endpoint is up and running before running the ``02_test_endpoint`` notebook