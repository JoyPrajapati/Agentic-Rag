# 🧠 Enterprise Knowledge Assistant (Agentic RAG)

A production-grade, multi-agent RAG system for querying financial 10-K filings. Built with **PydanticAI**, **LlamaIndex**, and **ChromaDB** — no Docker required.

## 🎯 What Makes This Different

Unlike traditional RAG (embed → search → generate), this system uses **6 specialized agents** with self-correction loops:



User Query
↓
[Planner] → Analyzes intent + complexity
↓
[Retriever] → Hybrid search (dense + BM25 + RRF)
↓
[Reranker] → Cross-encoder precision filter
↓
[Quality Gate] → Scores relevance, triggers fallback
↓
[Context Manager] → Trims to 6K token budget
↓
[Synthesizer] → Grounded answer with citations
↓
[Critique] → Validates faithfulness, loops back if failed
↓
Final Answer

----------------------------------------------------

## ✨ Features

- 🔍 **Hybrid retrieval** — dense (BGE-large) + sparse (BM25) with Reciprocal Rank Fusion
- 🎯 **Cross-encoder reranking** — BAAI/bge-reranker-large for precision
- 🧠 **3-tier memory** — short-term (5 turns), working (token budget), long-term (vector DB)
- 🛡️ **Hallucination guard** — Quality Gate + Critique agent validate every answer
- 📊 **RAGAS evaluation** — Faithfulness, Answer Relevancy, Context Precision/Recall
- 🌐 **REST API** — FastAPI with `/v1/chat`, `/v1/health`, `/v1/ingest`
- 💬 **Web UI** — Streamlit interface
- 🚀 **One-command setup** — `python setup.py`

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Orchestration** | PydanticAI (type-safe, no CVEs) |
| **Retrieval** | LlamaIndex + ChromaDB |
| **Embeddings** | BAAI/bge-large-en-v1.5 (local) |
| **Reranker** | BAAI/bge-reranker-large (local) |
| **LLM** | DeepSeek V3 via OpenRouter (free) |
| **API** | FastAPI |
| **UI** | Streamlit |
| **Evaluation** | RAGAS |

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- An OpenRouter API key ([get one free](https://openrouter.ai/keys))

### One-Command Setup

```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag.git
cd agentic-rag
python setup.py