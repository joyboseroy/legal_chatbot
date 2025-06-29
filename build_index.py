from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

DATA_DIR = "data"
CHROMA_DIR = "chroma_legal_index"

# Load all .txt files
all_docs = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_DIR, filename), encoding="utf-8")
        docs = loader.load()
        # Add filename metadata
        for doc in docs:
            doc.metadata["source"] = filename
        all_docs.extend(docs)

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in Chroma
vectorstore = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

vectorstore.persist()
print("✅ Chroma vector index built and saved.")
