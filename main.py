import os
from playwright.sync_api import sync_playwright
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from fastapi import FastAPI
from pydantic import BaseModel

def get_site_text_playwright(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(2000)
        text = page.text_content("body")
        browser.close()
        return text.strip() if text else ""

# -- CACHE TEXT TO FILE --
if os.path.exists("makonetworks_text.txt"):
    with open("makonetworks_text.txt") as f:
        site_text = f.read()
else:
    site_text = get_site_text_playwright("https://makonetworks.com")
    with open("makonetworks_text.txt", "w") as f:
        f.write(site_text)

print(f"Site text length: {len(site_text)}")

# -- CACHE VECTORDATABASE --
chroma_path = "./mako_chroma"
if os.path.exists(chroma_path) and os.listdir(chroma_path):

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(site_text)
else:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(site_text)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=chroma_path
    )

print(f"Chunks: {len(chunks)}")

def rag_query(question, db, llm_model="llama3.2:1b"):
    results = db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"You are a helpful assistant.\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    llm = OllamaLLM(
        model=llm_model,
        base_url="https://cxf-ollama-dev.cxfabric.io"
    )
    return llm.invoke(prompt)

app = FastAPI()
class ChatRequest(BaseModel):
    question: str

@app.post("/api/rag")
def rag_api(req: ChatRequest):
    answer = rag_query(req.question, db)
    return {
        "choices": [{
            "message": {
                "content": answer
            }
        }]
    }
