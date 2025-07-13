from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import requests
from bs4 import BeautifulSoup
import os

def get_site_text(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

DB_DIR = "./makonetworks_chroma"
if not os.path.exists(DB_DIR):
    site_text = get_site_text("https://makonetworks.com")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(site_text)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    db.persist()
else:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: ChatRequest):
    query = req.question
    results = db.similarity_search(query, k=3)
    retrieved = [doc.page_content for doc in results]
    prompt = f"""You are a helpful assistant.
Context:
{retrieved[0]}

{retrieved[1]}

{retrieved[2]}

Question: {query}
Answer:"""
    llm = Ollama(model="llama3")
    answer = llm.invoke(prompt)
    # Return as OpenAI-compatible JSON
    return {
        "choices": [{
            "message": {
                "content": answer
            }
        }]
    }
