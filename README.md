# resume-qa-rag
# 🤖 Resume RAG QA Bot

This project builds a **Retrieval-Augmented Generation (RAG)** system using 🧠 LangChain + HuggingFace FLAN-T5 to **ask questions from your resume** like a smart recruiter!

## 🔍 Features
- Upload your PDF resume
- Ask questions like "What are my projects?" or "What is my education?"
- Uses FAISS for vector search
- Uses `google/flan-t5-large` as the local LLM (no OpenAI key needed)

## 📸 Demo Screenshot
> ![screenshot](assets/demo.png)

## ⚙️ Tech Stack
- LangChain
- HuggingFace Transformers
- FAISS
- PyMuPDF
- Google Colab / Python

## 🚀 How to Run
1. Clone the repo
2. Install requirements
   ```bash
   pip install -r requirements.txt
