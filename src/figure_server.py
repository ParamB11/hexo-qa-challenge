import os
import json
import uvicorn
import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import traceback

import fitz  # PyMuPDF
import base64

# Load environment variables from .env file
load_dotenv()

# Configuration mapping based on the API reference
DATA_DIR = os.getenv("DATA_DIR")
VISION_MODEL = os.getenv("VISION_MODEL", "google/gemma-3-27b-it")
# VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
with open("logs/vllm_server_info.txt", "r") as f:
    vllm_url = f.read().strip()
VLLM_BASE_URL = vllm_url
PORT = int(os.getenv("FIGURE_SERVER_PORT", 8003))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
DEVICE = "cuda" # os.getenv("SENTENCE_TRANSFORMERS_DEVICE", "cpu")

# Set a custom cache directory to bypass the home directory disk quota
CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "../hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

# The API reference specifies CLIP for the figure server
# EMBEDDING_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_MODEL_NAME = "openai/clip-vit-large-patch14"

app = FastAPI(title="Datasheet Figure Retrieval Server")

# Initialize the embedding model globally
try:
    print(f"Loading CLIP model {EMBEDDING_MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE, cache_folder=CACHE_DIR)
except Exception as e:
    print(f"Failed to load embedding model: {e}")
    model = None

# Initialize the vLLM client (OpenAI compatible)
vllm_client = OpenAI(
    api_key="EMPTY", # vLLM doesn't require a real key by default
    base_url=f"{VLLM_BASE_URL}/v1",
)

# ---------------------------------------------------------
# Pydantic Models for Requests and Responses
# ---------------------------------------------------------
class RetrieveRequest(BaseModel):
    query: str
    datasheet_id: str
    top_k: Optional[int] = DEFAULT_TOP_K

class RetrieveResponse(BaseModel):
    query: str
    datasheet_id: str
    method: str
    response: str

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
        "index_type": "datasheet_figure_vision",
        "data_dir": DATA_DIR,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vision_model": VISION_MODEL,
        "vllm_base_url": VLLM_BASE_URL,
        "cache_dir": CACHE_DIR
    }

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_figures(request: RetrieveRequest):
    """Retrieve and interpret figures/diagrams using vision LLM."""
    if model is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
        
    datasheet_dir = os.path.join(DATA_DIR, request.datasheet_id)
    faiss_path = os.path.join(datasheet_dir, "figure_index.faiss")
    metadata_path = os.path.join(datasheet_dir, "figure_index.metadata.json")

    # 1. Check if index files exist for this datasheet
    if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Missing figure index for datasheet_id={request.datasheet_id}. Expected {faiss_path} and {metadata_path}"
        )

    try:
        # 2. Load FAISS index and metadata
        index = faiss.read_index(faiss_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # 3. Generate embedding for the text query using CLIP
        query_embedding = model.encode([request.query], normalize_embeddings=True).astype(np.float32)

        # --- MODIFICATION: Safeguard against dimension mismatch ---
        index_dim = index.d
        query_dim = query_embedding.shape[1]
        
        if index_dim != query_dim:
            error_msg = (
                f"Dimension mismatch! The FAISS index was built with vectors of dimension {index_dim}, "
                f"but the current model '{EMBEDDING_MODEL_NAME}' outputs dimension {query_dim}. "
                f"Please update EMBEDDING_MODEL_NAME to match the model originally used to create the index."
            )
            print(f"ERROR: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        # --------------------------------------------------------

        # 4. Perform search
        top_k = min(request.top_k, 50) # Max 50 as per API reference
        scores, indices = index.search(query_embedding, top_k)

        # --- DEBUGGING PRINTS ---
        print(f"\n--- DEBUG INFO ---")
        print(f"FAISS indices returned: {indices[0]}")
        print(f"Metadata type: {type(metadata)}, length: {len(metadata)}")
        if len(metadata) > 0:
            if isinstance(metadata, list) and isinstance(metadata[0], dict):
                print(f"Sample metadata keys (first item): {list(metadata[0].keys())}")
            elif isinstance(metadata, dict):
                print(f"Metadata is a dictionary. Sample keys: {list(metadata.keys())[:5]}")
        print(f"------------------\n")

        # 5. Extract top retrieved images (assuming base64 format in metadata)
        # retrieved_images = []
        # for idx in indices[0]:
        #     if idx != -1 and idx < len(metadata):
        #         # We assume the metadata stores the image as a base64 string or a local path.
        #         # If it's a base64 string:
        #         img_data = metadata[idx].get("image_base64") 
        #         if img_data:
        #             retrieved_images.append(img_data)

        # 5. Extract top retrieved images
        # retrieved_images = []
        # for idx in indices[0]:
        #     # FAISS returns -1 if there are no more neighbors found
        #     if idx == -1:
        #         print(f"DEBUG: Skipping idx {idx} (FAISS returned -1, no more neighbors).")
        #         continue
            
        #     # Check bounds if metadata is a list
        #     if isinstance(metadata, list) and idx >= len(metadata):
        #         print(f"DEBUG: Skipping idx {idx} (Out of bounds for metadata of length {len(metadata)}).")
        #         continue
            
        #     # Handle list vs dict indexing safely
        #     if isinstance(metadata, dict):
        #         # JSON dictionary keys are always strings
        #         item = metadata.get(str(idx))
        #     else:
        #         item = metadata[idx]

        #     if not item:
        #         print(f"DEBUG: Skipping idx {idx} (Item not found in metadata).")
        #         continue

        #     # Attempt to extract the image
        #     img_data = item.get("image_base64") 
            
        #     if img_data:
        #         retrieved_images.append(img_data)
        #         print(f"DEBUG: Successfully extracted image for index {idx}.")
        #     else:
        #         print(f"DEBUG: Failed at idx {idx}. 'image_base64' is missing or empty. Available keys in this item: {list(item.keys())}")

        # 5. Extract top retrieved images dynamically from PDF
        retrieved_images = []
        pdf_path = os.path.join(datasheet_dir, "datasheet.pdf")
        
        for idx in indices[0]:
            if idx != -1 and idx < len(metadata):
                item = metadata[idx]
                page_num = item.get("page_number")
                
                # Check if we have a valid page number and the PDF exists
                if page_num is not None and os.path.exists(pdf_path):
                    try:
                        doc = fitz.open(pdf_path)
                        # Metadata page numbers are usually 1-indexed, PyMuPDF is 0-indexed
                        target_page = int(page_num) - 1 
                        
                        if 0 <= target_page < len(doc):
                            page = doc[target_page]
                            # Render the page as an image (dpi=150 is usually a good balance of quality/size)
                            pix = page.get_pixmap(dpi=150)
                            img_data = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                            retrieved_images.append(img_data)
                            print(f"DEBUG: Successfully rendered page {page_num} from PDF.")
                            break # We only need the top 1 image for the LLM
                        
                        doc.close()
                    except Exception as e:
                        print(f"DEBUG: Failed to render PDF page: {e}")

        if not retrieved_images:
            return RetrieveResponse(
                query=request.query,
                datasheet_id=request.datasheet_id,
                method="vision",
                response="No relevant figures found in the datasheet."
            )

        # 6. Call the Vision LLM (vLLM running Gemma 3 27B)
        system_prompt = "You are an expert electrical engineer. Analyze the provided datasheet figure and answer the user's query."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Query: {request.query}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{retrieved_images[0]}" 
                        }
                    }
                ]
            }
        ]

        llm_response = vllm_client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.1
        )

        final_interpretation = llm_response.choices[0].message.content

        return RetrieveResponse(
            query=request.query,
            datasheet_id=request.datasheet_id,
            method="vision",
            response=final_interpretation
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Retrieval error: {repr(e)}")
    
    # try:
    #     # 2. Load FAISS index and metadata
    #     index = faiss.read_index(faiss_path)
    #     with open(metadata_path, "r", encoding="utf-8") as f:
    #         metadata = json.load(f)
    #     print(f"Loaded FAISS index and metadata for datasheet_id={request.datasheet_id}. Number of figures: {len(metadata)}")
    #     # 3. Generate embedding for the text query using CLIP
    #     query_embedding = model.encode([request.query], normalize_embeddings=True).astype(np.float32)
    #     print(f"Generated embedding for query: '{request.query}'")
    #     # 4. Perform search
    #     top_k = min(request.top_k, 50) # Max 50 as per API reference
    #     scores, indices = index.search(query_embedding, top_k)

    #     # 5. Extract top retrieved images (assuming base64 format in metadata)
    #     retrieved_images = []
    #     for idx in indices[0]:
    #         if idx != -1 and idx < len(metadata):
    #             # We assume the metadata stores the image as a base64 string or a local path.
    #             # If it's a base64 string:
    #             img_data = metadata[idx].get("image_base64") 
    #             if img_data:
    #                 retrieved_images.append(img_data)
    #     print(f"Retrieved {len(retrieved_images)} images for query: '{request.query}'")

    #     if not retrieved_images:
    #         return RetrieveResponse(
    #             query=request.query,
    #             datasheet_id=request.datasheet_id,
    #             method="vision",
    #             response="No relevant figures found in the datasheet."
    #         )

    #     # 6. Call the Vision LLM (vLLM running Gemma 3 27B)
    #     # We pass the top matched image to the vision model for interpretation
    #     system_prompt = "You are an expert electrical engineer. Analyze the provided datasheet figure and answer the user's query."
    #     print(f"Calling vision LLM with query: '{request.query}'")
    #     messages = [
    #         {"role": "system", "content": system_prompt},
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": f"Query: {request.query}"},
    #                 # Passing the top-1 image for interpretation to save context window/time
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{retrieved_images[0]}" 
    #                     }
    #                 }
    #             ]
    #         }
    #     ]

    #     llm_response = vllm_client.chat.completions.create(
    #         model=VISION_MODEL,
    #         messages=messages,
    #         max_tokens=1024,
    #         temperature=0.1
    #     )

    #     print(f"Received response from vision LLM for query: '{request.query}'")
    #     final_interpretation = llm_response.choices[0].message.content

    #     return RetrieveResponse(
    #         query=request.query,
    #         datasheet_id=request.datasheet_id,
    #         method="vision",
    #         response=final_interpretation
    #     )

    # except Exception as e:
    #     traceback.print_exc()
    #     raise HTTPException(status_code=500, detail=f"Retrieval error: {repr(e)}")

if __name__ == "__main__":
    print(f"Started server process on port {PORT}")
    # uvicorn.run(app, host="127.0.0.1", port=PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)