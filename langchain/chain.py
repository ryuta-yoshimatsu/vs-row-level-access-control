import os
from typing import Dict, List, Optional
import mlflow
from operator import itemgetter
from databricks.vector_search.client import VectorSearchClient
from langchain.schema.runnable import RunnableLambda
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_core.runnables import ConfigurableField
from langchain_core.documents.base import Document

config_path = os.path.join(os.getcwd(), "../../config.yaml")
config = mlflow.models.ModelConfig(development_config=config_path)


def extract_user_query_string(input: Dict) -> str:
    """
    Extracts the user's query string from the input data.

    Args:
        input: The input data containing user messages.

    Returns:
        The content of the last user message.
    """
    return input["messages"][-1]["content"]


def create_configurable_with_filters(input: Dict) -> Dict:
    """
    create configurable object with filters.

    Args:
        input: The input data containing filters.

    Returns:
        A configurable object with filters added to the search_kwargs.
    """
    configurable = {
        "configurable": {
            "search_kwargs": {
                "k": 2,
                "filters": input["filters"],
            }
        }
    }
    return configurable


def extract_pagecontent(docs: List[Document]) -> List[str]:
    """
    Extracts page content from a list of langchain documents.

    Args:
        docs: A list of documents.

    Returns:
        A list of page content extracted from the documents.
    """
    return [doc.page_content for doc in docs]


############
# Connect to the Vector Search Index
############
vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index(
    endpoint_name=config.get("vs_endpoint"), index_name=f'{config.get("catalog")}.{config.get("schema")}.{config.get("index_name")}'
)


############
# Turn the Vector Search index into a LangChain retriever
############
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column="chunk",
    columns=[
        "chunk",
    ],
).as_retriever()

############
# Convert the retriever into configurable retriever so that run time parameters such as filters 
# can be  passed down during chain invoke
############
configurable_vs_retriever = vector_search_as_retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)


############
# RAG Chain
############
chain_with_configurable = (
    {"messages": itemgetter("messages"), "filters": itemgetter("filters")}
    | RunnableLambda(
        lambda input: configurable_vs_retriever.invoke(
            extract_user_query_string(input), config=create_configurable_with_filters(input)
        )
    )
    | extract_pagecontent
)

## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=chain_with_configurable)
