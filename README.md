# CensusInsight AI — RAG-Powered Census Data Explorer

<div align="center">
  <h3>📺 Project Demo</h3>
  <a href="https://www.loom.com/share/04bb98b9676744878bc63823df403b8e">
    <img src="https://cdn.loom.com/sessions/thumbnails/04bb98b9676744878bc63823df403b8e-44be78ef2147f7c3-full-play.gif" alt="CensusInsight AI Demo" style="width:100%; max-width:600px; border-radius: 10px;">
  </a>
  <p><i>Click the image above to watch the full demo on Loom</i></p>
</div>


> **Portfolio Project** | LangChain · Groq · ChromaDB · Streamlit  
> Demonstrates production-ready Retrieval-Augmented Generation (RAG) using the LangChain Expression Language (LCEL) pipeline.

---

## 📌 What This Project Demonstrates

This app lets users query official U.S. Census PDF documents using plain English. It showcases the full RAG stack that employers look for in LLM/AI engineering roles:

| Concept                 | Implementation                                                           |
| ----------------------- | ------------------------------------------------------------------------ |
| **RAG Pipeline**        | PDF ingestion → chunking → embedding → vector retrieval → LLM generation |
| **LCEL Chain**          | Modern `RunnablePassthrough` + `\|` operator composition                 |
| **Vector Store**        | ChromaDB with persistent storage (skip re-indexing on reload)            |
| **Embeddings**          | OpenAI `text-embedding-ada-002` for semantic search                      |
| **Fast Inference**      | Groq API running `llama-3.1-8b-instant` for sub-second responses         |
| **Streaming**           | `st.write_stream()` for real-time token-by-token output                  |
| **Caching**             | `@st.cache_resource` to avoid redundant DB loads                         |
| **Source Transparency** | Sidebar shows retrieved chunks with page numbers                         |

---

## Architecture

```
User Query
    │
    ▼
[Streamlit UI]
    │
    ▼
[ChromaDB Retriever] ──── similarity search (k=3) ────▶ [Relevant Chunks]
    │                                                         │
    │                                                         ▼
    └──────────────────── LCEL Chain ──────────────────── [Prompt Template]
                                                              │
                                                              ▼
                                                        [Groq LLM / Llama 3.1]
                                                              │
                                                              ▼
                                                      [StrOutputParser]
                                                              │
                                                              ▼
                                                    Streamed Answer + Sources
```

**Data Indexing Flow (first run only):**

```
./data/*.pdf
    │
    ▼
PyPDFDirectoryLoader
    │
    ▼
RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
    │
    ▼
OpenAIEmbeddings → ChromaDB (persisted to ./chroma_db_census/)
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/censusinsight-ai
cd censusinsight-ai
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file:

```env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

Get your keys:

- **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys) (used for embeddings)
- **Groq** (free tier available): [console.groq.com](https://console.groq.com)

### 3. Add Census PDFs

```bash
mkdir data
# Add your Census Bureau PDFs here, e.g.:
# data/2020_census_summary.pdf
# data/demographic_profile.pdf
```

Free Census PDFs: [census.gov/library/publications](https://www.census.gov/library/publications.html)

### 4. Run the App

```bash
streamlit run app.py
```

The first run will index your PDFs into ChromaDB (~30 seconds). Subsequent runs load instantly from the persisted vector store.

---

## Requirements

```txt
streamlit
langchain
langchain-openai
langchain-groq
langchain-chroma
langchain-community
langchain-text-splitters
python-dotenv
pypdf
chromadb
```

---

## 💡 Key Design Decisions

**Why Groq instead of OpenAI for the LLM?**  
Groq's LPU hardware delivers ~10x faster inference than standard GPU APIs. For a chat interface, this means responses feel instantaneous. The `llama-3.1-8b-instant` model is also free-tier friendly, making this project easy to run without cost.

**Why OpenAI for embeddings but Groq for inference?**  
`text-embedding-ada-002` produces high-quality semantic embeddings that ChromaDB indexes once and persists. At query time, only a tiny embedding call is needed, the heavy lifting is done by Groq's fast inference.

**Why `chunk_size=1000` with `overlap=200`?**  
Census documents are dense with statistics. A 1000-token chunk captures a full statistical table or paragraph. The 200-token overlap ensures context isn't lost at chunk boundaries (e.g., a table header won't be separated from its rows).

**Why `temperature=0.1`?**  
Census data is factual. Low temperature keeps the model from hallucinating statistics or inventing demographic figures.

---

## 🔧 Possible Extensions

- [ ] Add metadata filtering (filter by year, state, demographic group)
- [ ] Multi-turn conversation memory with `ConversationBufferMemory`
- [ ] Hybrid search (BM25 + semantic) using `EnsembleRetriever`
- [ ] Evaluation pipeline with RAGAS for answer faithfulness scoring
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces

---

## 📚 LangChain Concepts Covered

- `langchain_core` LCEL (`|` operator, `RunnablePassthrough`)
- `ChatPromptTemplate` with context injection
- `StrOutputParser` for clean text output
- `Chroma` vector store with persistence
- `PyPDFDirectoryLoader` + `RecursiveCharacterTextSplitter`
- `as_retriever()` with `search_kwargs`
- Streaming with `.stream()`

---

## 🙋 About

Built as a portfolio demonstration of production LangChain patterns. The same RAG architecture scales to legal documents, financial reports, research papers, and internal knowledge bases.

_Feel free to fork, extend, or use as a reference for your own RAG projects._
