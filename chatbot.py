from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load Chroma vector DB
CHROMA_DIR = "chroma_legal_index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever()

# Use TinyLlama, Phi, or any local Ollama model
llm = Ollama(model="tinyllama")

# Custom prompt
legal_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=\"\"\"... (as above) ...\"\"\"
)

# Chain setup
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": legal_prompt_template},
    return_source_documents=True
)

# Function to expose
def ask_legal_bot(query):
    response = qa_chain.run(query)
    return response
