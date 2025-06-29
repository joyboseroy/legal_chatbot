# app.py (Final version with smart re-indexing, ChromaDB, multi-format support, and custom legal prompt)
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import hashlib
import pickle

# Directories
CHROMA_DIR = "chroma_legal_index"
DATA_DIR = "data"
HASH_FILE = os.path.join(CHROMA_DIR, "file_hashes.pkl")

# Utility to compute file hash
def compute_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

# Check for file changes
def has_files_changed():
    current_hashes = {
        fname: compute_file_hash(os.path.join(DATA_DIR, fname))
        for fname in os.listdir(DATA_DIR)
        if fname.lower().endswith((".txt", ".pdf", ".docx", ".html", ".htm"))
    }
    try:
        with open(HASH_FILE, 'rb') as f:
            previous_hashes = pickle.load(f)
        if current_hashes == previous_hashes:
            return False
    except FileNotFoundError:
        pass

    with open(HASH_FILE, 'wb') as f:
        pickle.dump(current_hashes, f)
    return True

# Rebuild index only if files changed
if has_files_changed():
    documents = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        elif filename.endswith(".pdf"):
            loader = PyMuPDFLoader(filepath)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath)
        elif filename.endswith(".html") or filename.endswith(".htm"):
            loader = UnstructuredHTMLLoader(filepath)
        else:
            continue

        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectorstore.persist()
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

retriever = vectorstore.as_retriever()

# Load LLM from Ollama
llm = Ollama(model="tinyllama")

# Custom legal prompt template
legal_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a legal assistant trained in Indian laws and procedures. You will answer based only on the information provided in the context below. Do not speculate or make up information. If the context does not contain an answer, say so politely.

Your tone should be:
- Professional
- Cautious
- Clear
- Easy to understand by a non-lawyer

---
### Context:
{context}

---
### Question:
{question}

---
### Answer:
"""
)

# Set up RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": legal_prompt_template},
    return_source_documents=True
)

# Streamlit UI setup
st.set_page_config(page_title="India Legal Chatbot (Offline)", layout="centered")
st.title("\U0001f1ee\U0001f1f3 India Legal Chatbot (Offline)")
st.markdown("Enter your question about Indian law. Your answer will be based on local legal documents (.txt, .pdf, .docx, .html) in the 'data' folder.")

# Main input for chatbot
user_input = st.text_input("Ask a legal question:")

if user_input:
    with st.spinner("Looking up legal context..."):
        response = qa_chain({"query": user_input})
        st.markdown("**Answer:**")
        st.write(response["result"])

        with st.expander("View source documents"):
            for doc in response["source_documents"]:
                st.markdown(f"📄 **{doc.metadata.get('source', 'Unknown Source')}**")
                st.markdown(doc.page_content[:500] + "...")
