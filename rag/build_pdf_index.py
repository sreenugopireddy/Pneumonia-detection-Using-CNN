from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import glob

print("Loading PDFs...")

pdfs = glob.glob("rag/data/*.pdf")

docs = []

for p in pdfs:
    print("Reading:", p)
    loader = PyPDFLoader(p)
    docs.extend(loader.load())

print("Splitting into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building FAISS index...")

db = FAISS.from_documents(chunks, embeddings)

db.save_local("rag/vector_db")

print("✅ PDF RAG index built successfully")
