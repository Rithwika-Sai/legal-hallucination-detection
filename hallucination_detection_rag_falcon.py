import os
import re
import json
import torch
import faiss
import numpy as np
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
# Using Falcon-7B-Instruct
FALCON_MODEL     = "tiiuae/falcon-7b-instruct"
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL        = "cross-encoder/nli-deberta-v3-small"
JSON_PATH        = "data/test.json"

TOP_K            = 2  # Reduced K for Falcon's smaller 2048 context window
HALLUC_THRESHOLD = 0.5
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SentenceResult:
    sentence: str
    top_evidence: str
    entailment_score: float
    is_hallucinated: bool

# ─── 1. DEEP SEARCH DATA LOADER ──────────────────────────────────────────────
print(f"[1/6] Loading {JSON_PATH}...")
DOCUMENTS = []

def extract_text_dynamically(obj):
    if isinstance(obj, str): return obj
    if isinstance(obj, dict):
        for k in ['text', 'content', 'document', 'judgment', 'body']:
            if k in obj and isinstance(obj[k], str): return obj[k]
        candidates = [v for v in obj.values() if isinstance(v, str)]
        return max(candidates, key=len) if candidates else None
    return None

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                text = extract_text_dynamically(item)
                if text: DOCUMENTS.append(text.replace("\n", " ").strip())
        except json.JSONDecodeError:
            print("Error: Invalid JSON format."); exit()
else:
    print(f"Error: {JSON_PATH} not found."); exit()

if not DOCUMENTS:
    print("CRITICAL ERROR: No documents loaded."); exit()

print(f"Successfully loaded {len(DOCUMENTS)} documents.")

# ─── 2. EMBEDDING & NLI MODELS ──────────────────────────────────────────────
print("[2/6] Loading Embedding & NLI Models...")
embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
nli = CrossEncoder(NLI_MODEL, device=DEVICE)

print("[3/6] Building FAISS Index...")
embeddings = embedder.encode(DOCUMENTS, convert_to_numpy=True)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype('float32'))

# ─── 3. LOAD FALCON (FP16 MODE) ─────────────────────────────────────────────
print("[4/6] Loading Falcon-7B (Float16)...")
tokenizer = AutoTokenizer.from_pretrained(FALCON_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Note: Falcon-7B requires ~15GB VRAM in Float16. 
# We avoid trust_remote_code=True as Falcon is now native in transformers.
model = AutoModelForCausalLM.from_pretrained(
    FALCON_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True
)

# ─── 4. RAG & GENERATION ────────────────────────────────────────────────────
def retrieve(query, k=TOP_K):
    q_vec = embedder.encode([query])
    _, indices = index.search(np.array(q_vec).astype('float32'), k=k)
    return [DOCUMENTS[i] for i in indices[0]]

def generate_answer(query, chunks):
    context = "\n".join([f"- {c}" for c in chunks])
    # Falcon specific prompt format (No INST tags)
    prompt = (
        f"Use the following context to answer the question. If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    
    # Falcon Context window is 2048
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.2, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return response

# ─── 5. HALLUCINATION DETECTION logic ──────────────────────────────────────
def verify_response(response):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if len(s.strip()) > 15]
    results = []
    
    for s in sentences:
        evidence = retrieve(s, k=1)
        # NLI Score (index 1 is entailment)
        scores = nli.predict([(evidence[0], s)], apply_softmax=True)
        score = float(scores[0][1])
        
        results.append(SentenceResult(
            sentence=s,
            top_evidence=evidence[0],
            entailment_score=round(score, 4),
            is_hallucinated=score < HALLUC_THRESHOLD
        ))
    return results

# ─── 6. EXECUTION ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = "Provide a summary of the main legal arguments."
    print(f"\n[5/6] Processing Query: {query}")
    
    chunks = retrieve(query)
    ans = generate_answer(query, chunks)
    
    print(f"\nFALCON RESPONSE:\n{ans}")
    print("\n[6/6] Verifying claims...")
    
    reports = verify_response(ans)
    print("\n" + "─"*60)
    print("FALCON HALLUCINATION REPORT:")
    for r in reports:
        flag = "🔴 HALLUCINATED" if r.is_hallucinated else "🟢 SUPPORTED"
        print(f"\n{flag} (Score: {r.entailment_score})")
        print(f"Sentence: {r.sentence}")
    print("─"*60)