from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS

persist_directory = "db"

def ingest_documents():
    loader = None  # Initialize the loader variable before the loop
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))  # Update loader here
    if loader is None:
        print("No PDF files found.")
        return  # Exit if no PDF files were found

    documents = loader.load()
    print("splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    # create embeddings here
    print("Loading sentence transformers model")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # create vector store here
    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")

if __name__ == "__main__":
    ingest_documents()
