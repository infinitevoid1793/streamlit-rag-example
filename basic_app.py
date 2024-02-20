# Import the necessary modules
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_community.vectorstores import FAISS
import tempfile
import os
import time 
from langchain_community.document_loaders import PyPDFLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
import logging
from dotenv import load_dotenv
logging.basicConfig(level=logging.INFO)
from utils import FileIngestor, VectorDB, ChatBot


load_dotenv()
BEGIN_RAG_CONVERSATION = "Lets begin our conversation about the document you uploaded!"


if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def generic_conversation():
    pass 


def upload_file():
    uploaded_file = st.sidebar.file_uploader('Choose your .pdf file', type="pdf")
    if uploaded_file is not None and "FILE_UPLOADED" not in st.session_state:
        document,path = FileIngestor(uploaded_file).ingest_file()
        st.session_state.db = VectorDB(task="generate").embed_document(docs=document,path=path.rsplit("/",1)[-1])  
        success_message =st.sidebar.success("PDF file uploaded successfully!")
        time.sleep(2)
        success_message.empty()
        uploaded_file.close()      
        st.session_state.FILE_UPLOADED  = True 
    elif "FILE_UPLOADED" in st.session_state:
        if st.session_state.FILE_UPLOADED:
            pass 
        else:
            st.warning("Please upload a file to continue.")





def sidebar():
    llm_type = st.sidebar.selectbox('Language Model', ["OpenAI", "Llama2"])
    if llm_type == "OpenAI":
        api_key_location = st.sidebar.selectbox('API Key', [".env", "Input Manually"])
        if api_key_location == "Input Manually":
            api_key = st.sidebar.text_input("API Key",type="password")
        elif api_key_location == ".env":
            api_key = os.getenv("OPENAI_API_KEY")
    elif llm_type == "Llama2":
        model_path = st.sidebar.text_input("Model Path", "./models/<model>.gguf")

    st.session_state.ACTION = st.sidebar.selectbox('Action', ["Generic Conversation", "Retrieval Q&A"])
    if st.session_state.ACTION == "Retrieval Q&A":
        upload_file()

def main_page(action: str = "Generic Conversation"):
    title_init=st.title("Hello, I'm InFVoid, your assistant!")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if st.session_state.ACTION == "Generic Conversation":
        if "LLM" not in st.session_state:
            st.session_state.LLM=ChatBot(action=st.session_state.ACTION)
        generic_conversation()
    if st.session_state.ACTION == "Retrieval Q&A" and "FILE_UPLOADED" in st.session_state:
        if "LLM" not in st.session_state:
            st.session_state.LLM=ChatBot(action=st.session_state.ACTION)
        st.session_state.LLM_CHAIN = ConversationalRetrievalChain.from_llm(
                                                                st.session_state.LLM.llm,
                                                                st.session_state.db.as_retriever(search_kwargs={'k': 2}),
                                                                    return_source_documents=False
                                                                )
            
        if prompt := st.chat_input("Enter queries here."):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            ## Implement chat response here 
            response = st.session_state.LLM_CHAIN.invoke({'question': prompt, 'chat_history': st.session_state.chat_history[-5:]}).get('answer')
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Define the Streamlit app
def main():  
    sidebar()
    main_page()

# Run the Streamlit app
if __name__ == "__main__":
    main()