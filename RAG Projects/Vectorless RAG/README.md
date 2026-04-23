# Vectorless RAG System

This project implements a Retrieval-Augmented Generation (RAG) system without using vector databases.

## 🔧 Tech Stack
- Python
- scikit-learn (TF-IDF)
- OpenAI API

## 🚀 Features
- Keyword-based retrieval using TF-IDF
- Context-based answer generation
- Lightweight (no vector DB required)

## 📂 Files
- vectorless_rag.py → Main code
- emaar_properties.txt → Dataset

## ▶️ How to Run

1. Install dependencies:
pip install openai scikit-learn

2. Set API key:
setx OPENAI_API_KEY "your_key"

3. Run:
python vectorless_rag.py