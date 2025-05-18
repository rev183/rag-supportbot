import os

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai.llms import OpenAI
from scraper import scrape_angelone_support_pages


def load_docs():
    loader = PyPDFDirectoryLoader("sources")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return text_splitter.split_documents(docs)

def load_angelone_texts():
    scrape_angelone_support_pages()
    text_loader = DirectoryLoader("sources/angelone-support", glob="**/*.txt", loader_cls=TextLoader)
    text_docs = text_loader.load()
    text_text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    return text_text_splitter.split_documents(text_docs)

def init_rag():
    embeddings = OpenAIEmbeddings()
    documents = load_docs()
    documents.extend(load_angelone_texts())
    db = Chroma.from_documents(documents, embeddings)
    retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    llm = OpenAI(temperature=0)
    # rag_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="map_rerank",
    #     retriever=retriever,
    #     return_source_documents=True,
    #     verbose=True,
    # )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,
    )
    return conversation_chain


