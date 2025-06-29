# legal_chatbot
A simple legal chatbot you can install on your local laptop

Here’s what this project looks like at the end:

A chatbot UI powered by Streamlit
Powered by a local LLM like TinyLlama via Ollama
Reads .pdf, .docx, .html, and .txt legal files
Answers user queries using a retrieval-based QA pipeline
Shows relevant law sources for every answer
Smart re-indexing — updates only when your files change
(All tools are open source and running locally)

This is perfect for:

Law students
Civic tech activists
Anyone building tools for access to justice
Step 1: Setting Up Your Local Stack
Let’s start with the basics. You’ll need:

Requirements
A laptop (Windows/Linux/Mac) with at least 8GB RAM
Python 3.10+
Comfort with the command line
Install Python Libraries
pip install langchain chromadb streamlit sentence-transformers unstructured[docx] python-docx pymupdf
Install Ollama and a Local LLM
Ollama is an amazing tool for running large language models on your own machine.

# Install from https://ollama.com
ollama run tinyllama
You can also experiment with models like phi, llama2, or mistral if your machine allows.

Step 2: Prepare Your Legal Data
Create a folder called data/. This is where your legal documents will go. Think of it like your law library.

Example files:

hindu_succession_act.pdf
model_tenancy_act.txt
senior_citizen_law.docx
consumer_protection.html
The more readable and structured these files are, the better the chatbot will perform.

You can get high-quality versions by searching in google or from:

India Code
eGazette
Government websites for state-specific rent acts or succession rules
Step 3: Indexing the Law
This part is a bit technical, but extremely powerful. We’re going to use LangChain to process and split documents, Chroma as a vector database, and sentence-transformers to create embeddings.

And we’ll add smart re-indexing so you don’t waste time reloading the same files.

# core logic in app.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os, hashlib, pickle
CHROMA_DIR = "chroma_legal_index"
DATA_DIR = "data"
HASH_FILE = os.path.join(CHROMA_DIR, "file_hashes.pkl")
# Check if files changed
...
# Load and split
...
# Embed and store
...
Step 4: Making It Conversational
Now comes the magic: turning it into a chatbot.

We use Streamlit to create the UI. It’s clean, fast, and perfect for rapid prototypes.

We use a custom legal prompt to give it the right tone:

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a legal assistant trained in Indian laws. Only answer using the context below. If unsure, say so politely.
Context:
{context}
Question:
{question}
Answer:
"""
)
And then connect it with:

qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="tinyllama"),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
Step 5: Run and Use
You’re ready. From your terminal:

ollama run tinyllama
streamlit run app.py
Visit localhost:8501 and try asking:

What happens if someone dies without a will?
Can a tenant be evicted without notice?
What rights do senior citizens have under Indian law?
The model will scan your documents and reply with a confident, context-based answer. And you can even see which file it came from.

