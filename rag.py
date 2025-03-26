from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

# =======================
# Funções auxiliares
# =======================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")  # Diretório para persistência do índice
embeddings = OpenAIEmbeddings()

def load_documents():
    """
    Função para carregar os documentos a partir do diretório.
    """
    doc_files = ["banco.txt", "graficos.txt", "manual_do_usuario.txt", "documentacao.txt", "links.txt"]
    docs = []
    
    for file_name in doc_files:
        loader = TextLoader(os.path.join(DOCS_DIR, file_name), encoding="utf-8")
        docs.extend(loader.load())
    
    return docs

def create_faiss_index():
    """
    Função para criar o índice FAISS a partir dos documentos.
    """
    docs = load_documents()
    if os.path.exists(INDEX_DIR):
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        print("Índice FAISS carregado do diretório local.")
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(INDEX_DIR)
        print("Índice FAISS criado e salvo localmente.")
    
    return vectorstore

def get_relevant_context(query, k=20):
    """
    Função para recuperar o contexto relevante de uma query utilizando FAISS.
    """
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    relevant_docs = vectorstore.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    return context