import os
import json
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
MODEL_ID    = "mistralai/Mistral-7B-Instruct-v0.2"
EMBED_MODEL = "all-MiniLM-L6-v2"
JSON_PATH   = "data/test.json"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 1. DEEP SEARCH LOAD ───────────────────────────────────────────────────
print(f"[1/4] Loading {JSON_PATH}...")
DOCUMENTS = []

def extract_text_dynamically(obj):
    """Recursively search for the largest string value in a JSON object."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        # Sort keys to try and find 'text' or 'body' first, but eventually take the longest string
        candidates = [v for v in obj.values() if isinstance(v, str)]
        if candidates:
            return max(candidates, key=len)
    return None

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
    print("CRITICAL ERROR: No documents were loaded. Please check if your JSON contains text fields.")
    exit()

print(f"Successfully loaded {len(DOCUMENTS)} documents.")

# ─── 2. INDEXING (FAISS) ──────────────────────────────────────────────────
print("[2/4] Initializing Vector Index...")
embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
embeddings = embedder.encode(DOCUMENTS, convert_to_numpy=True)

# Ensure embeddings is 2D before indexing
if len(embeddings.shape) == 1:
    embeddings = embeddings.reshape(1, -1)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype('float32'))

# ─── 3. LOAD MISTRAL ──────────────────────────────────────────────────────
print("[3/4] Loading Mistral-7B-Instruct (FP16)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True
)

# ─── 4. RAG PIPELINE ──────────────────────────────────────────────────────
def ask_mistral_rag(query):
    q_embed = embedder.encode([query])
    # Retrieve top 3
    k = min(3, len(DOCUMENTS))
    _, indices = index.search(np.array(q_embed).astype('float32'), k=k)
    
    context_parts = [DOCUMENTS[i] for i in indices[0]]
    context_string = "\n\n".join(context_parts)

    prompt = (
        f"<s>[INST] Use the context to answer. If unsure, say you don't know.\n\n"
        f"Context:\n{context_string}\n\n"
        f"Question: {query} [/INST]"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text.split("[/INST]")[-1].strip()

# ─── EXECUTION ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[4/4] System Ready.")
    query = "Give me a summary of the provided data."
    print(f"\nQUERY: {query}")
    print(f"MISTRAL RESPONSE:\n{ask_mistral_rag(query)}")

