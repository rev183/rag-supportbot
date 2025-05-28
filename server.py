import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware
from models import ChatResponse, QueryRequest
from rag_helper import init_rag

logger = logging.getLogger("server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('App starting up')
    yield
    # Clean up the ML models and release the resources
    logger.info('App shutting down')

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

conversation_chain = init_rag()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    """
    Receives a user query, processes it using the RAG chain, and returns the answer
    along with source documents.
    """
    if conversation_chain is None:
        raise HTTPException(status_code=503, detail="RAG components not initialized.")

    try:
        # Invoke the RAG chain with the user's query
        result = conversation_chain.invoke({"question": request.query})

        # Extract answer and source documents
        answer = result.get("answer", "I Don't know")
        source_docs = result.get("source_documents", [])

        # Format source documents for the response
        formatted_sources = []
        for doc in source_docs:
            formatted_sources.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

        return ChatResponse(answer=answer, source_documents=formatted_sources)

    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# --- Health Check Endpoint (Optional) ---
@app.get("/")
async def read_root():
    return {"message": "FastAPI RAG backend is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8882, env_file=".env")