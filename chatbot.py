import time
import streamlit as st
import azure_openai_util
import config
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv

load_dotenv()


def load_vector_store(vector_store_folder_path, vectore_store_name):
    vector_store = FAISS.load_local(
        folder_path=vector_store_folder_path, 
        embeddings=azure_openai_util.embeddings(), 
        index_name=vectore_store_name,
        allow_dangerous_deserialization=True)
    return vector_store 


def generateChain():
    llm = azure_openai_util.load_llm()
    # chain -> take the question, get relevant document, pass it to the LLM, generate the output
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain


def generate_stream(response):
    # generate the stream of messages
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05)

not_found_response = """
I cannot help, please ask questions about JBL devices (JBL Pulse 5, JBL Clip 5, JBL Charge 5 Wi-Fi, JBL Bar 500, JBL Bar 5.0 MultiBeam)
"""

def main():
    st.set_page_config(
        page_title="JBL Products Chatbot App",
        page_icon="🧊",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    st.header("JBL Products Chatbot")
    vector_store = load_vector_store(config.vector_store_folder_path, config.vector_store_index_name)
    chain = generateChain()

    with st.chat_message("assistant"):
        st.markdown("Ask me anything about JBL devices (JBL Pulse 5, JBL Clip 5, JBL Charge 5 Wi-Fi,JBL Bar 500, JBL Bar 5.0 MultiBeam)")

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
