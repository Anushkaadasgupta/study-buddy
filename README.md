# Study Buddy AI

##  Project Overview
Study Buddy AI is an Agentic AI-based assistant designed to help students understand Artificial Intelligence concepts. It uses Retrieval-Augmented Generation (RAG), memory, and tools to provide accurate and context-based answers.

---

##  Objective
To build an intelligent assistant that:
- Answers questions from AI study material
- Avoids hallucination
- Maintains conversation memory
- Uses tools for real-time queries

---

##  Features
-  RAG-based question answering (ChromaDB)
-  Memory using LangGraph (multi-turn conversation)
-  Tool support (date/time queries)
-  Streamlit web interface
-  Self-evaluation node for response quality

---

##  Tech Stack
- Python
- LangGraph
- ChromaDB
- Sentence Transformers
- Streamlit

---

##  Project Structure
study-buddy/
│
├── agent.py
├── dataset.py
├── capstone_streamlit.py
├── requirements.txt
└── README.md

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run capstone_streamlit.py
