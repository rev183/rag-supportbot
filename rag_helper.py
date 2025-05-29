import os
import traceback
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai.llms import OpenAI
from dotenv import load_dotenv
from scraper import scrape_angelone_support_pages

load_dotenv()
OPENAI_GEN_LLM = OpenAI(temperature=0.5)

def generate_questions_for_chunk(doc: Document) -> Document:
    print(f"Generating questions for chunk from source: {doc.metadata.get('source', 'N/A')}")
    prompt = f"""
    You are an expert at generating concise questions that can be answered by the provided text.
    Generate 3-5 diverse questions that can be answered *solely* from the following text chunk.
    Make sure the questions are clear and directly related to the content.
    Return only the questions, one per line, without numbering or introductory phrases.

    Text:
    ---
    {doc.page_content}
    ---

    Questions:
    """
    try:
        response = OPENAI_GEN_LLM.invoke(prompt)
        generated_questions = [q.strip() for q in response.split('\n') if q.strip()]

        if generated_questions:
            questions_text = "\n\n--- Possible Questions for this Document ---\n"
            questions_text += "\n".join(generated_questions)
            doc.page_content += questions_text
            print(f"Generated {len(generated_questions)} questions for a chunk.")
        else:
            print("No questions generated for this chunk.")

    except Exception as e:
        print(f"Error generating questions for a chunk: {e}")
        traceback.print_exc()

    return doc

def load_and_process_docs():
    loader = PyPDFDirectoryLoader("sources")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_pdf_docs = text_splitter.split_documents(docs)
    processed_pdf_docs = [generate_questions_for_chunk(doc) for doc in split_pdf_docs]
    return processed_pdf_docs

def load_angelone_texts():
    scrape_angelone_support_pages()
    text_loader = DirectoryLoader("sources/angelone-support", glob="**/*.txt", loader_cls=TextLoader)
    text_docs = text_loader.load()
    text_text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    return text_text_splitter.split_documents(text_docs)

def init_rag():
    embeddings = OpenAIEmbeddings()
    documents = load_and_process_docs()
    documents.extend(load_angelone_texts())
    db = Chroma.from_documents(documents, embeddings)
    retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    llm = OpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,
    )
    return conversation_chain


