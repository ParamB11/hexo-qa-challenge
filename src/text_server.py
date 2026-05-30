import os
import json
import uvicorn
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration mapping based on the API reference
DATA_DIR = os.getenv("DATA_DIR")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
PORT = int(os.getenv("TEXT_SERVER_PORT", 8001))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
DEVICE = os.getenv("SENTENCE_TRANSFORMERS_DEVICE", "cpu")

app = FastAPI(title="Datasheet Text Retrieval Server")

# Initialize the embedding model globally
try:
    print(f"Loading embedding model {EMBEDDING_MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

# ---------------------------------------------------------
# Pydantic Models for Requests and Responses
# ---------------------------------------------------------
class RetrieveRequest(BaseModel):
    query: str
    datasheet_id: str
    top_k: Optional[int] = DEFAULT_TOP_K

class ResultItem(BaseModel):
    text: str
    score: float

class RetrieveResponse(BaseModel):
    query: str
    datasheet_id: str
    method: str
    results: List[ResultItem]
    num_results: int

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    """Health check endpoint returning server status."""
    if model is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    return {
        "status": "healthy",
        "index_type": "datasheet_text_dense",
        "data_dir": DATA_DIR,
        "embedding_model": EMBEDDING_MODEL_NAME
    }

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_text(request: RetrieveRequest):
    """Retrieve relevant text chunks from the specified datasheet."""
    if model is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
        
    datasheet_dir = os.path.join(DATA_DIR, request.datasheet_id)
    faiss_path = os.path.join(datasheet_dir, "text_index.faiss")
    metadata_path = os.path.join(datasheet_dir, "text_index.metadata.json")

    # 1. Check if index files exist for this datasheet
    if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Missing text index for datasheet_id={request.datasheet_id}. Expected {faiss_path} and {metadata_path}"
        )

    try:
        # 2. Load FAISS index and metadata
        index = faiss.read_index(faiss_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # 3. Generate embedding for the query
        # For E5 models, it is recommended to prefix queries with "query: "
        query_text = f"query: {request.query}" if "e5" in EMBEDDING_MODEL_NAME.lower() else request.query
        query_embedding = model.encode([query_text], normalize_embeddings=True).astype(np.float32)

        # 4. Perform search
        top_k = min(request.top_k, 50) # Max 50 as per API reference
        scores, indices = index.search(query_embedding, top_k)

        # 5. Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(metadata):
                results.append(ResultItem(
                    text=metadata[idx].get("text", ""),
                    score=float(score)
                ))

        return RetrieveResponse(
            query=request.query,
            datasheet_id=request.datasheet_id,
            method="dense_text",
            results=results,
            num_results=len(results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

if __name__ == "__main__":
    print(f"Started server process on port {PORT}")
    # uvicorn.run(app, host="127.0.0.1", port=PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)