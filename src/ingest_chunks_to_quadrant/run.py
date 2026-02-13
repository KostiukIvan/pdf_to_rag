import json
import os
import uuid
import re
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURATION
# ==========================================
QDRANT_URL = "http://localhost:6333" 
COLLECTION_NAME = "insurance_docs"
INPUT_FILE = "docs/md/ready/chunks_for_qdrant.jsonl"

# The REAL name of your document (Hardcoded or derived)
REAL_DOC_NAME = "RIBOs-Consolidated-Examinee-Resource-September-18-2024.pdf"

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384 

print(f"Loading local model: {MODEL_NAME}...")
embedding_model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(url=QDRANT_URL)

def clean_metadata(chunk: Dict) -> Dict:
    """
    Transforms raw chunk data into clean, citation-ready metadata.
    Example: 
      Input:  {'source_file': 'page_0005.md', 'page_num': 5}
      Output: {'source_file': 'RIBOs...pdf', 'page_num': 5, 'citation': 'RIBOs...pdf (Page 5)'}
    """
    # 1. extract page number safely if not present
    if 'page_num' not in chunk:
        # Try to parse from filename "page_0005.md"
        match = re.search(r"page_(\d+)", chunk.get('source_file', ''))
        if match:
            chunk['page_num'] = int(match.group(1))
    
    # 2. Overwrite the filename with the REAL document name
    chunk['original_filename'] = chunk.get('source_file') # Keep old one for debugging
    chunk['source_file'] = REAL_DOC_NAME
    
    # 3. Create a helpful "Citation String" for the LLM to use later
    chunk['citation'] = f"{REAL_DOC_NAME} (Page {chunk.get('page_num', '?')})"
    
    return chunk

def create_deterministic_id(text: str, source: str, page: int) -> str:
    unique_string = f"{source}_{page}_{text}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

def ingest_data():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
    
    if not os.path.exists(INPUT_FILE):
        print(f"No input file found at {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} chunks to ingest.")
    
    batch_size = 64
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i : i + batch_size]
        
        current_batch_payloads = []
        texts_to_embed = []
        ids = []

        for line in batch_lines:
            try:
                raw_chunk = json.loads(line)
                
                # --- APPLY METADATA CLEANING HERE ---
                chunk = clean_metadata(raw_chunk)
                
                # 1. Text to Embed: "Topic: Content"
                # This helps the vector model find the right context
                searchable_text = f"{chunk.get('topic', 'General')}: {chunk.get('text', '')}"
                texts_to_embed.append(searchable_text)
                
                # 2. ID: Use the REAL source name now
                point_id = create_deterministic_id(chunk['text'], chunk['source_file'], chunk.get('page_num', 0))
                ids.append(point_id)

                # 3. Payload: The clean metadata
                current_batch_payloads.append(chunk)
                
            except json.JSONDecodeError:
                continue

        if texts_to_embed:
            vectors = embedding_model.encode(texts_to_embed).tolist()
            
            points = [
                PointStruct(id=uid, vector=vec, payload=pay)
                for uid, vec, pay in zip(ids, vectors, current_batch_payloads)
            ]

            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"  -> Ingested batch {i} to {i+len(batch_lines)}")

    print("✅ Ingestion Complete!")

if __name__ == "__main__":
    ingest_data()