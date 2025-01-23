import os
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import config

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


def get_all_files(directory):
    path = Path(os.path.join(os.getcwd(), directory))
    if not path.exists():
        raise Exception(f"Directory {path.absolute()} does not exist")
    return [str(file) for file in path.rglob("*") if file.is_file()]
    

def process_pdfs(pdfs_directory, vector_store_folder_path, vector_store_index_name):
    print(f"Processing PDF documents from directory `{pdfs_directory}`")
    pdf_files = get_all_files(pdfs_directory)
    text = ""
    for pdf_file in pdf_files:
        # Read pdf file
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # #Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings())
    vector_store.save_local(vector_store_folder_path, vector_store_index_name)
    print(f"Finished Processing PDF documents to vector store path `{vector_store_folder_path}`")

if __name__ == "__main__":
    process_pdfs(config.pdfs_directory, config.vector_store_folder_path, config.vector_store_index_name)