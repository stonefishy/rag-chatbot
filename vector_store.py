from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import azure_openai_util
import config

def save_vector_store(pdf_path, vector_store_folder_path, vector_store_index_name):
    # Read pdf file
    pdf_reader = PdfReader(pdf_path)
    text = ""
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
    print(f"Converting PDF document and save embeddings to vector store path `{vector_store_folder_path}`")


if __name__ == "__main__":
    save_vector_store(config.pdf_path, config.vector_store_folder_path, config.vector_store_index_name)