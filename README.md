# RAG-Based Document Question Answering System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that enables users to ask natural language questions about a PDF document and receive context-aware answers.

The system combines:

* Semantic document retrieval using FAISS
* Sentence Transformer embeddings
* Local LLM inference using Ollama
* Interactive chatbot interface using Streamlit

The objective is to reduce hallucinations by providing the language model with relevant context retrieved directly from the source document before generating an answer.

---

# Project Architecture

```text
PDF Document
      │
      ▼
PyPDFLoader
      │
      ▼
Document Chunking
(500 chars, 100 overlap)
      │
      ▼
SentenceTransformer
(all-MiniLM-L6-v2)
      │
      ▼
FAISS Vector Database
      │
      ▼
Top-K Semantic Retrieval
(K = 8)
      │
      ▼
Retrieved Context
      │
      ▼
Mistral LLM (Ollama)
      │
      ▼
Generated Answer
      │
      ▼
Streamlit Chat Interface
```

---

# Project Structure

```text
RAG/
│
├── app.py
├── streamlit_app.py
│
├── data/
│   └── Agmlo.pdf
│
├── faiss_store/
│   ├── faiss.index
│   └── metadata.pkl
│
└── src/
    ├── data_loader.py
    ├── embedding.py
    ├── vectorstore.py
    └── search.py
```

---

# Components

## 1. Document Loading

File:

```text
src/data_loader.py
```

Purpose:

* Loads PDF documents using PyPDFLoader.
* Extracts page content.
* Preserves metadata for future retrieval.

Library Used:

```python
PyPDFLoader
```

---

## 2. Document Chunking

File:

```text
src/embedding.py
```

Chunk Configuration:

```python
chunk_size = 500
chunk_overlap = 100
```

Why Chunking?

Large language models cannot process entire documents efficiently.

The document is divided into smaller overlapping chunks so that:

* Context is preserved
* Retrieval becomes more accurate
* Relevant sections can be found quickly

Example:

```text
Chunk 1:
Payment Services...

Chunk 2:
Services...
Refund Policy...
```

The overlap prevents important information from being split across chunk boundaries.

---

## 3. Embedding Generation

Embedding Model:

```text
all-MiniLM-L6-v2
```

Library:

```python
sentence-transformers
```

Why This Model?

Advantages:

* Lightweight
* Fast inference
* Good semantic understanding
* Suitable for local deployment

Output Dimension:

```text
384
```

Each chunk becomes a 384-dimensional vector representation.

---

## 4. Vector Database

File:

```text
src/vectorstore.py
```

Vector Database:

```text
FAISS
```

Purpose:

* Stores chunk embeddings
* Performs fast similarity search
* Retrieves semantically relevant chunks

Search Method:

```python
IndexFlatL2
```

Top Retrieval Count:

```python
top_k = 8
```

---

## 5. Retrieval-Augmented Generation

File:

```text
src/search.py
```

Process:

1. User asks a question.
2. Question converted into embedding.
3. FAISS retrieves top matching chunks.
4. Retrieved chunks combined as context.
5. Context sent to Mistral model.
6. Model generates final answer.

Prompt Template:

```text
You are a document question-answering assistant.

Answer the user's question directly using only the context.

Do NOT summarize the entire context.

Extract only the information that answers the question.

If the answer is not found, say:
"I could not find that information in the document."
```

This prompt helps reduce hallucinations.

---

## 6. Language Model

LLM Used:

```text
Mistral (via Ollama)
```

Model:

```text
mistral:latest
```

Advantages:

* Fully local execution
* No API cost
* Good reasoning ability
* Suitable for RAG systems

---

# Streamlit Features

The chatbot interface provides:

### Natural Language Queries

Users can ask questions directly.

Example:

```text
What is the refund policy for buyers?
```

---

### Streaming Responses

Answers are generated token-by-token for a conversational experience.

---

### Source Passage Display

The retrieved chunks used to generate the answer are displayed.

This improves transparency and explainability.

---

### System Information Sidebar

Displays:

* Current model
* Indexed chunk count

---

### Chat Reset

Allows users to clear the conversation history.

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd RAG
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Windows:

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Ollama Setup

Install Ollama:

https://ollama.com

Pull Mistral:

```bash
ollama pull mistral
```

Verify:

```bash
ollama list
```

Expected:

```text
mistral:latest
```

---

# Building the Vector Database

Run:

```bash
python app.py
```

On first run:

1. PDF is loaded.
2. Chunks are created.
3. Embeddings generated.
4. FAISS index created.

Generated Files:

```text
faiss_store/faiss.index
faiss_store/metadata.pkl
```

Future runs load the existing index automatically.

---

# Running the Chatbot

Terminal Version:

```bash
python app.py
```

---

Streamlit Version:

```bash
streamlit run streamlit_app.py
```

---

# Sample Queries

### Query 1

```text
Compare payment services and refund policy for buyers?
```
<img width="1355" height="652" alt="Screenshot 2026-06-04 164151" src="https://github.com/user-attachments/assets/90d88997-f8c1-46a0-ad3c-5bcdcfad779a" />

Result:

System retrieves payment-related chunks and generates a focused answer.

---

### Query 2

```text
What is the refund policy for buyers?
```
<img width="1340" height="634" alt="Screenshot 2026-06-04 163220" src="https://github.com/user-attachments/assets/acc4c005-d500-4b8e-8f0e-cf10d708bf48" />

Result:

System extracts refund-related information and summarizes it.

---

### Query 3

what is ebay?
 <img width="1336" height="658" alt="Screenshot 2026-06-04 162027" src="https://github.com/user-attachments/assets/18574825-d608-46ab-ba2e-6228b8c03bb8" />



---

### Query 4

```text
Give me a short overview of ebay's payment system.
```
<img width="1347" height="650" alt="Screenshot 2026-06-04 163832" src="https://github.com/user-attachments/assets/8c617aaf-0549-415a-b872-b98285ab08f4" />

Result:

Here it is.

---

### Query 5 (Failure Case)

```text
Who is eBay wife?
```
<img width="1360" height="652" alt="Screenshot 2026-06-04 165257" src="https://github.com/user-attachments/assets/59b5eea8-91e3-4e78-a451-0c85631c2a8d" />

Result:

```text
I could not find that information in the document.
```

This demonstrates that the system avoids answering questions not supported by the source document.

---

# Limitations

### Retrieval Dependency

Answer quality depends on retrieved chunks.

If relevant chunks are not retrieved, answer quality decreases.

---

### Hallucination Risk

Although reduced through RAG, hallucinations may still occur in rare situations.

---

### Broad Queries

Questions such as:

```text
Tell me everything about eBay
```

may retrieve multiple unrelated chunks and produce less focused answers.

---

### Performance

Running locally on CPU can result in slower response times compared to GPU deployments.

---

# Future Improvements

Potential enhancements:

* Hybrid Search (BM25 + Vector Search)
* Reranking Models
* Multi-PDF Support
* Metadata Filtering
* Conversation Memory
* Better Embedding Models (BGE, E5)
* Citations with Page Numbers
* Docker Deployment

---

# Technologies Used

* Python
* Streamlit
* FAISS
* Sentence Transformers
* LangChain
* Ollama
* Mistral
* NumPy
* PyPDFLoader

---

