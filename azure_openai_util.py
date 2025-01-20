import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.chat_models import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_key = os.getenv("OPENAI_API_KEY")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")


def embeddings():
    # generating embedding
    return AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment,
        api_version=api_version,
        api_key=api_key,
        azure_endpoint=azure_endpoint)


def load_llm():
    # define the LLM
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=chat_deployment,
        temperature=0,
        max_tokens=1000,
    )

    return llm
