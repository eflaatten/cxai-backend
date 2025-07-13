from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import requests
from bs4 import BeautifulSoup

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.post("/api/rag")
async def rag_api(req: ChatRequest):
    # 1. Fetch site text (runs each request, OK for demo)
    r = requests.get("https://makonetworks.com")
    s = BeautifulSoup(r.text, "html.parser")
    for tag in s(["script", "style", "noscript"]): tag.decompose()
    text = s.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    site_text = "\n".join(lines)

    # 2. Chunk & embed (in-memory, OK for demo)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(site_text)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="https://cxf-ollama-dev.cxfabric.io/v1/chat/completions")
    db = Chroma.from_texts(texts=chunks, embedding=embeddings)

    # 3. Retrieve and generate answer
    results = db.similarity_search(req.question, k=3)
    retrieved = [doc.page_content for doc in results]
    prompt = f"""You are a helpful assistant.
Context:
{retrieved[0] if len(retrieved)>0 else ''}

{retrieved[1] if len(retrieved)>1 else ''}

{retrieved[2] if len(retrieved)>2 else ''}

Question: {req.question}
Answer:"""

    llm = Ollama(model="llama3.2:1b", base_url="https://cxf-ollama-dev.cxfabric.io/v1/chat/completions")
    answer = llm.invoke(prompt)
    return JSONResponse({
        "choices": [{
            "message": {
                "content": answer
            }
        }]
    })
