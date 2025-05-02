import os
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
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
