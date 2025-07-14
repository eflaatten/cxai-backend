import json
from playwright.sync_api import sync_playwright
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma

def get_site_text_playwright(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(2000)
        text = page.text_content("body")
        browser.close()
        return text.strip() if text else ""

site_text = get_site_text_playwright("https://makonetworks.com")
print(f"Site text length: {len(site_text)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(site_text)
print(f"Chunks: {len(chunks)}")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./mako_chroma"
)

def rag_query(question, db, llm_model="llama3.2:1b"):
    results = db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"You are a helpful assistant.\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    llm = OllamaLLM(
    model=llm_model,
    base_url="https://cxf-ollama-dev.cxfabric.io"
    )
    return llm.invoke(prompt)

if __name__ == "__main__":
    question = "Does Mako Networks work with oil/gas companies?"
    answer = rag_query(question, db)
    print("Q:", question)
    print("A:", answer)
