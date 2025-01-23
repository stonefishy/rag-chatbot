import time
import os
import streamlit as st
import config
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings


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


def load_vector_store(vector_store_folder_path, vectore_store_name):
    vector_store = FAISS.load_local(
        folder_path=vector_store_folder_path, 
        embeddings=embeddings(), 
        index_name=vectore_store_name,
        allow_dangerous_deserialization=True)
    return vector_store 


def generateChain():
    llm =  AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=chat_deployment,
        temperature=0,
        max_tokens=1000,
    )
    # chain -> take the question, get relevant document, pass it to the LLM, generate the output
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain


def generate_stream(response):
    # generate the stream of messages
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05)


def main():
    st.set_page_config(
        page_title="JBL Products Chatbot",
        page_icon="🧊",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    st.header("JBL Products Chatbot")
    vector_store = load_vector_store(config.vector_store_folder_path, config.vector_store_index_name)
    chain = generateChain()

    with st.chat_message("assistant"):
        st.markdown("Ask me anything about JBL devices (JBL Pulse 5, JBL Clip 5, JBL Bar 500)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me something"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = "I am thinking..."
        with st.spinner(response):
            match_documents = vector_store.similarity_search(prompt)
            response = chain.run(input_documents = match_documents, question = prompt)

        with st.chat_message("assistant"):
            st.write_stream(generate_stream(response))
        st.session_state.messages.append({"role": "assistant", "content": response})


main()
