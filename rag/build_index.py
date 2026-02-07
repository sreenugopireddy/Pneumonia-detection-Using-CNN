import sys, os
sys.path.append(os.path.dirname(__file__))

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(__file__)
KNOW_DIR = os.path.join(BASE_DIR, "knowledge")
OUT_DIR = os.path.join(BASE_DIR, "vector_db")

docs = []

for fname in os.listdir(KNOW_DIR):
    path = os.path.join(KNOW_DIR, fname)
    loader = TextLoader(path, encoding="utf-8")
    docs.extend(loader.load())

print("Loaded docs:", len(docs))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)
db.save_local(OUT_DIR)

print("Vector DB built at:", OUT_DIR)
