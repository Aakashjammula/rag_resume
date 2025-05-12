import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# — Load env vars (on Vercel these come from the dashboard) —
load_dotenv()

# — Initialize Pinecone & index —
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "resume-index"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
pinecone_index = pc.Index(index_name)

# — Embeddings + vectorstore —
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings)

# — Prompt template + LLM —
prompt = PromptTemplate.from_template("""
You are an intelligent assistant. Use only the Context below.

Context:
{context}

Question:
{question}

Answer:
""".strip())

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)

# — FastAPI app setup —
app = FastAPI()

class Query(BaseModel):
    q: str
    k: int = 5

@app.post("/ask")
async def ask(req: Query):
    if not req.q:
        raise HTTPException(status_code=400, detail="Missing question text")
    docs = vectorstore.similarity_search(req.q, k=req.k)
    context = "\n\n".join(d.page_content for d in docs)
    user_prompt = prompt.invoke({"context": context, "question": req.q})
    resp = llm.invoke(user_prompt)
    return {
        "context": context,
        "answer": getattr(resp, "content", str(resp)),
    }
