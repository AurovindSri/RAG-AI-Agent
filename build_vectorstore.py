##################Included code snippets with and without QDrant, With and without GPU for convenience ##########################

"""import os
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Folder containing .txt and .pdf files
DOCS_FOLDER = "Input_Files"  # Replace with your folder path

# 1. Load all .txt and .pdf files
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            continue  # Skip unsupported file types
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    return documents

# 2. Main process
def build_and_save_vector_store():
    print("Loading documents...")
    raw_docs = load_documents_from_folder(DOCS_FOLDER)

    if not raw_docs:
        print("No documents loaded. Check folder path and file formats.")
        return

    print(f"Loaded {len(raw_docs)} documents. Splitting...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
    chunks = splitter.split_documents(raw_docs)

    print(f"Creating embeddings for {len(chunks)} chunks...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)

    vector_store.save_local("vector_index")
    print("Vector store saved to 'vector_index/'")

if __name__ == "__main__":
    build_and_save_vector_store()
"""


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Folder containing .txt and .pdf files
DOCS_FOLDER = "Input_Files"
COLLECTION_NAME = "document_vectors"

def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            continue  # Skip unsupported file types
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    return documents

def build_and_save_vector_store():
    print("Loading documents...")
    raw_docs = load_documents_from_folder(DOCS_FOLDER)

    if not raw_docs:
        print("No documents loaded. Check folder path and file formats.")
        return

    print(f"Loaded {len(raw_docs)} documents. Splitting...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
    chunks = splitter.split_documents(raw_docs)

    print(f"Creating embeddings for {len(chunks)} chunks...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Connect to Qdrant (ensure Qdrant is running at this host/port)
    client = QdrantClient(host="localhost", port=6333)

    # Check and create collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}'.")
    else:
        print(f"Using existing Qdrant collection '{COLLECTION_NAME}'.")

    print("Storing vectors in Qdrant...")
    vector_store = Qdrant.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url="http://localhost:6333",  # use `url` instead of passing `client`
        collection_name=COLLECTION_NAME,
    )

    print(f"Vector store saved in Qdrant collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    build_and_save_vector_store()

"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

# Folder containing .txt and .pdf files
DOCS_FOLDER = "Input_Files"
COLLECTION_NAME = "document_vectors"

# GPU-compatible embedding wrapper
class GPUHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to("cuda")  # move model to GPU

    def embed_documents(self, texts):
        return [self.model.encode(text, convert_to_tensor=True).cpu().numpy() for text in texts]

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True).cpu().numpy()

    def __call__(self, text):  # <- Required for compatibility
        return self.embed_query(text)

# Load .txt and .pdf documents from folder
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            continue
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    return documents

# Main vector store build function
def build_and_save_vector_store():
    print("Loading documents...")
    raw_docs = load_documents_from_folder(DOCS_FOLDER)

    if not raw_docs:
        print("No documents loaded. Check folder path and file formats.")
        return

    print(f"Loaded {len(raw_docs)} documents. Splitting...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
    chunks = splitter.split_documents(raw_docs)

    print(f"Creating embeddings on GPU for {len(chunks)} chunks...")
    embedding_model = GPUHuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    client = QdrantClient(host="localhost", port=6333)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}'.")
    else:
        print(f"Using existing Qdrant collection '{COLLECTION_NAME}'.")

    print("Storing vectors in Qdrant...")
    vector_store = Qdrant.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url="http://localhost:6333",
        collection_name=COLLECTION_NAME,
    )

    print(f"Vector store saved in Qdrant collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    build_and_save_vector_store()

