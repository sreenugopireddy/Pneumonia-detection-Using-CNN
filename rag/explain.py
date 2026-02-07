import sys, os
sys.path.append(os.path.dirname(__file__))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(__file__)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    os.path.join(BASE_DIR, "vector_db"),
    embeddings,
    allow_dangerous_deserialization=True
)

def get_explanation(label):

    docs = db.similarity_search(label, k=2)

    context = "\n".join([d.page_content for d in docs])

    return f"""
Prediction: {label}

Medical Explanation:
{context}
"""
