import os
import json
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
MODEL_ID    = "tiiuae/falcon-7b-instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
JSON_PATH   = "data/test.json"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 1. DEEP SEARCH LOAD ───────────────────────────────────────────────────
print(f"[1/4] Loading {JSON_PATH}...")
DOCUMENTS = []

def extract_text_dynamically(obj):
    """Recursively search for the best string content in a JSON object."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        # Look for typical keys first, otherwise take the longest string found
        for key in ['text', 'content', 'document', 'body', 'judgment']:
            if key in obj and isinstance(obj[key], str):
                return obj[key]
        candidates = [v for v in obj.values() if isinstance(v, str)]
        if candidates:
            return max(candidates, key=len)
    return None
#ingestion
if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                text = extract_text_dynamically(item)
                if text and len(text.strip()) > 10:
                    DOCUMENTS.append(text.replace("\n", " ").strip())
        except json.JSONDecodeError:
            print("Error: test.json is not a valid JSON file.")
            exit()
else:
    print(f"Error: {JSON_PATH} not found.")
    exit()

if not DOCUMENTS:
    print("CRITICAL ERROR: No documents were loaded. Check your JSON format.")
    exit()

print(f"Successfully loaded {len(DOCUMENTS)} documents.")

print("[2/4] Initializing Vector Index...")
#vectorization
embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
embeddings = embedder.encode(DOCUMENTS, convert_to_numpy=True)

if len(embeddings.shape) == 1:
    embeddings = embeddings.reshape(1, -1)
#indexing faiss
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype('float32'))

# ─── 3. LOAD FALCON-7B ────────────────────────────────────────────────────
print("[3/4] Loading Falcon-7B (FP16)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Note: Removed trust_remote_code to use the latest stable library version
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True
)

# ─── 4. RAG PIPELINE ──────────────────────────────────────────────────────
#ret and aug
def ask_falcon_rag(query):
    # Retrieve top relevant snippets
    #retrieval
    q_embed = embedder.encode([query])
    k = min(2, len(DOCUMENTS)) # Falcon has a smaller 2048 window; keep context tight
    _, indices = index.search(np.array(q_embed).astype('float32'), k=k)
    #augmentation
    context_string = "\n\n".join([DOCUMENTS[i] for i in indices[0]])

    # Falcon Instruct Prompt Format
    prompt = (
        f"Answer the question using only the context provided. If the answer is not there, say you don't know.\n\n"
        f"Context:\n{context_string}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    
    # Token safety: Truncate to 2048 (Falcon's limit)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    #generation
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract only the part after "Answer:"
    return full_text.split("Answer:")[-1].strip()

# ─── EXECUTION ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[4/4] System Ready.")
    query = "Summarize the key information in the document."
    print(f"\nQUERY: {query}")
    print("-" * 30)
    print(f"FALCON RESPONSE:\n{ask_falcon_rag(query)}")
    print("-" * 30)