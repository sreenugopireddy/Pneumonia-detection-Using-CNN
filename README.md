# 🫁 Pneumonia Detection Using CNN + Explainable AI + PDF-RAG Assistant

An end-to-end AI system that detects pneumonia from chest X-ray images using a Convolutional Neural Network with transfer learning, provides visual explanations using Grad-CAM, and generates evidence-grounded medical explanations using a PDF-based Retrieval Augmented Generation (RAG) pipeline with a chatbot interface.

This project goes beyond basic image classification and demonstrates a full **Explainable Medical AI Assistant prototype**.

---

# 🚀 Features

✅ Chest X-ray pneumonia detection (CNN + Transfer Learning)  
✅ MobileNetV2 fine-tuned on medical dataset  
✅ Grad-CAM heatmap explainability  
✅ Confidence calibration + thresholding  
✅ PDF-grounded RAG medical explanation  
✅ Medical evidence chatbot (PDF knowledge base)  
✅ Streamlit diagnostic dashboard UI  
✅ Batch-ready design  
✅ Clinical-style PDF report generator (optional module)

---

# 🏗 Tech Stack

**ML / AI**
- PyTorch
- TorchVision
- MobileNetV2 (transfer learning)
- Grad-CAM

**RAG / NLP**
- LangChain
- FAISS vector database
- Sentence-Transformers embeddings
- PDF semantic retrieval

**App Layer**
- Streamlit

**Data**
- Chest X-ray Pneumonia Dataset (Kaggle)

---

# 📊 Model Details

- Backbone: **MobileNetV2**
- Strategy: Transfer learning
- Fine-tuning: Last block + classifier head
- Loss: Weighted CrossEntropy (class imbalance handling)
- Input size: 224×224
- Output: Binary classification (NORMAL / PNEUMONIA)

---

# 📈 Evaluation Results (Test Set)**

---

# 🔍 Explainability Layer

Grad-CAM is used to visualize which lung regions influenced the model decision.

Outputs:
- Heatmap overlay
- Model attention regions
- Visual trust signal

This supports explainable AI requirements in medical imaging.

---

# 📚 PDF-Grounded RAG System

Instead of generic AI text, explanations are retrieved from indexed medical PDFs:

Examples:
- Pneumonia fact sheets
- Prevention guidelines
- Treatment references
- Antibiotic usage notes

Pipeline:


PDF → text chunks → embeddings → FAISS index → semantic search → answer

Chatbot answers are **evidence-grounded**, not hallucinated.

---

# 🖥 Streamlit App UI

Dashboard includes:

- X-ray viewer
- Grad-CAM overlay
- Prediction + confidence
- Risk indicator
- Medical explanation
- Sidebar medical chatbot

Run:

---

# ⚙️ Installation

## 1️⃣ Clone Repo

git clone https://github.com/sreenugopireddy/Pneumonia-detection-Using-CNN.git
cd Pneumonia-detection-Using-CNN


## 2️⃣ Create Environment

python -m venv venv
venv\Scripts\activate


## 3️⃣ Install Dependencies

pip install torch torchvision streamlit
pip install grad-cam
pip install langchain langchain-community langchain-text-splitters
pip install sentence-transformers faiss-cpu
pip install pypdf


---

# ⚠️ Limitations

- Dataset is pediatric — domain shift affects external images
- Class imbalance causes higher false positives
- Model is a screening aid — not a diagnosis tool
- Softmax confidence is not clinical probability
- External X-ray styles may reduce accuracy

---

# 🧪 Future Improvements

- Larger balanced dataset
- Multi-class lung disease detection
- DICOM support
- Multimodal RAG (image + text retrieval)
- Similar case search
- Clinical report export automation
- Model calibration curves

---

# 📌 Disclaimer

This system is for **educational and research purposes only**.  
It is **not a medical diagnostic device**.

---

# 👤 Author

**Sreenu GopiReddy**  
AI / ML Project — Medical Imaging + RAG Assistant

---

# ⭐ If You Found This Useful

Star the repo and share feedback!
