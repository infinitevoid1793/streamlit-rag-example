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
from langchain.callbacks.manager import CallbackManager
import logging

logging.basicConfig(level=logging.INFO)

class TextSplitter():
    def __init__(self,type: str = "recursive"):
        self.splitter = self.get_splitter(type)

    def get_splitter(self,type: str = "recursive") -> str:
        if type == "recursive":
            return RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
        elif type == "html":
            return HTMLHeaderTextSplitter()
        else:
            raise ValueError("Splitter not supporter") 
    

class VectorDB():

    def __init__(self,task: str = 'generate',llm_type: str = "OpenAI"):
        self.task = task
        self._llm = llm_type

    def _get_embeddings(self):
        if self._llm == 'OpenAI':
            return OpenAIEmbeddings()
        else:
            return HuggingFaceEmbeddings()
        
    @staticmethod    
    def remove_newlines(document):
        document.page_content =document.page_content.replace("\n", " ")
        return document

    def embed_document(self, docs: list = [],path: str = "",save: bool = False):
        embeddings = self._get_embeddings()
        text_splitter = TextSplitter().splitter
        split_docs = [self.remove_newlines(document) for document in text_splitter.split_documents(docs)]
        db = FAISS.from_documents(split_docs,embeddings)
        if save:
            db.save_local("db_{}".format(path))
        return db

    def _similarity_search(self, query: str):
        pass 

    def with_retriever(self):
        return self.db.as_retriever() 



class ChatBot():
    def __init__(self,model: str = "OpenAI",action: str = "Generic Conversation",model_path: str = None,streaming : bool = True):
        logging.info("Initializing ChatBot")
        self.llm = self._init_llm(model) 
        self.model_path = model_path

    @staticmethod
    def _init_llm(model: str = "OpenAI"):
        if model == "OpenAI":
            return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,streaming=True,callbacks=CallbackManager([StreamingStdOutCallbackHandler()]))
        if model == "HuggingFace":
            return None 
    def echo_response(self, user_input: str) -> str:
        return f"Echo: {user_input}"



class FileIngestor:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        self._file_path = None
        self._loader = None

    def ingest_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            self._file_path = tmp_file.name
            self._loader = PyPDFLoader(file_path=self._file_path)
            logging.info("Loading Document: {}".format(self._file_path))
        return (self._loader.load(),self._file_path)