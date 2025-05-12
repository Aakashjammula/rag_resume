import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load env
load_dotenv()

# —— Init Pinecone & LangChain ——
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
idx = "resume-index"
if not pc.has_index(idx):
    pc.create_index(name=idx, dimension=384, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
pinecone_index = pc.Index(idx)

embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings)

prompt = PromptTemplate.from_template("""
You are an intelligent assistant. Use only the Context below.

Context:
{context}

Question:
{question}

Answer:
""".strip())

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)

# —— FastAPI setup ——
app = FastAPI()

class Query(BaseModel):
    q: str
    k: int = 5

@app.post("/ask")
async def ask(req: Query):
    if not req.q:
        raise HTTPException(400, "Missing question text")
    docs = vectorstore.similarity_search(req.q, k=req.k)
    context = "\n\n".join(d.page_content for d in docs)
    user_prompt = prompt.invoke({"context": context, "question": req.q})
    resp = llm.invoke(user_prompt)
    return {
        "context": context,
        "answer": getattr(resp, "content", str(resp)),
    }

# —— Uvicorn entry point ——
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
