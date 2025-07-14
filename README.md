# Mako Networks RAG: ChromaDB + Ollama

This project is a minimal Retrieval-Augmented Generation (RAG) pipeline for querying content from [makonetworks.com](https://makonetworks.com) using ChromaDB for vector search and Ollama for LLM-powered answers.

---

## What does the script do?

The main script (`main.py`) performs the following steps:

1. **Scrapes makonetworks.com** using Playwright to collect up-to-date website content.
2. **Splits the scraped text** into manageable chunks for processing.
3. **Embeds the content** using Ollama embedding models (e.g., `nomic-embed-text`).
4. **Stores and indexes embeddings** in a local ChromaDB database for fast vector search.
5. **Answers user queries** by retrieving relevant context from ChromaDB and generating answers with an Ollama LLM (e.g., `llama3` (`llama3.2:1b` required for k8) ).

---

## Setup Instructions

### Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) (for local embeddings/LLM)
  - Models: `nomic-embed-text` and `llama3`
- [Playwright](https://playwright.dev/python/) (for dynamic site scraping)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/eflaatten/cxai-backend.git
   cd cxai-backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m playwright install
   # Or manually:
   pip install langchain langchain-ollama chromadb playwright requests beautifulsoup4
   python -m playwright install
   ```

3. **[Local Ollama Only] Pull required models:**
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3/Preferred Model
   ```

---

The script will:
- Scrape makonetworks.com for content (using a headless browser)
- Chunk, embed, and index content with ChromaDB

---

## Using a Remote Ollama Server

If your Ollama model is running remotely (e.g., in a Kubernetes cluster or on a remote server):

1. Edit the `rag_query` function in `main.py`:
   ```python
   def rag_query(question, db, llm_model="llama3"):
       ...
       llm = OllamaLLM(
           model=llm_model,
           base_url="https://YOUR-OLLAMA-ENDPOINT"
       )
       return llm.invoke(prompt)
   ```

---

**Testing Locally**

1. Run 
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

2. In Postman, make a POST request to `http://0.0.0.0:8000/api/rag` with the following JSON body:
```json
{
    "question": "What is Mako Networks?"
}
```

---

**You should receive a response similar to:**
```json
{
"choices": [
        {
            "message": {
                "content": "Mako Networks is another name for the Mako System."
            }
        }
    ]
}
```
