import os
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import azure_openai_util
import config


def get_all_files(directory):
    path = Path(os.path.join(os.getcwd(), directory))
    if not path.exists():
        raise Exception(f"Directory {path.absolute()} does not exist")
    return [str(file) for file in path.rglob("*") if file.is_file()]
    

def convert_pdfs_to_vector_store(pdfs_directory, vector_store_folder_path, vector_store_index_name):
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
    vector_store = FAISS.from_texts(chunks, azure_openai_util.embeddings())
    vector_store.save_local(vector_store_folder_path, vector_store_index_name)
    print(f"Converting PDF documents to vector store path `{vector_store_folder_path}`")

if __name__ == "__main__":
    convert_pdfs_to_vector_store(config.pdfs_directory, config.vector_store_folder_path, config.vector_store_index_name)