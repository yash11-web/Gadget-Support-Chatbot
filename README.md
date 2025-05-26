# Gadget FAQ Chatbot

This project implements an LLM-based **Retrieval-Augmented Generation (RAG)** chatbot to automate customer support for a company that sells electronic gadgets such as smartphones, laptops, and accessories.

---

##  Problem Statement

The customer support team is overwhelmed with repetitive inquiries such as:

- Product specifications (smartphones, laptops, accessories)
- Order tracking
- Return policies
- Payment methods
- Warranty information

Currently, these are handled manually. The goal is to **automate responses** to these frequently asked questions (FAQs) using a chatbot powered by a small but capable LLM (`microsoft/phi-1_5` from Hugging Face) and an FAISS vector store for efficient retrieval.

---

## Features

-  **Natural language question answering**
-  **Multi-turn conversation support** with context
-  **Context-aware answers** based on a provided FAQ knowledge base
-  **Local vector database (FAISS)** for fast retrieval
-  **Streamlit Web UI**
-  Built using `LangChain`, `Transformers`, `HuggingFace`, and `FAISS`

