import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE = os.path.dirname(__file__)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    os.path.join(BASE, "vector_db"),
    embeddings,
    allow_dangerous_deserialization=True
)


# -------- Single Best Evidence --------

def rag_answer(query):

    doc = db.similarity_search(query, k=1)[0]

    src = os.path.basename(doc.metadata.get("source", "PDF"))

    return f"""
Source: {src}

{doc.page_content}
"""


# -------- Model Explanation Builder --------

def build_prediction_explanation(label, confidence):

    if label == "PNEUMONIA":
        query = "pneumonia symptoms diagnosis treatment chest xray"
    else:
        query = "normal chest xray characteristics clear lungs"

    evidence = rag_answer(query)

    return f"""
## 📘 AI Medical Explanation

**Prediction:** {label}  
**Confidence:** {confidence:.2f}

{evidence}

---
Answer retrieved from indexed medical PDF.
"""
