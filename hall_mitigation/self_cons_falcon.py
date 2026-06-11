import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np

# 1. SETUP MODELS
# Note: Falcon-H1R-7B is a top choice in 2026 for high-speed reasoning
model_id = "tiiuae/Falcon-H1R-7B" 
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Ensure padding is set for batch generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True # Required for Falcon's custom architecture
)

# Semantic model for voting
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def generate_falcon_consistency(prompt, num_samples=5):
    """
    Generates multiple reasoning paths using Falcon and returns the consensus.
    """
    # Falcon-style instruction format
    formatted_prompt = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    print(f"Falcon is generating {num_samples} parallel reasoning paths...")
    
    # --- STEP 1: MULTI-PATH GENERATION ---
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,         # Crucial for path diversity
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode all candidates
    candidates = []
    for i in range(num_samples):
        # We strip the prompt text from the output
        gen_text = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        candidates.append(gen_text)

    # --- STEP 2: SEMANTIC MAJORITY VOTE ---
    embeddings = embedder.encode(candidates, convert_to_tensor=True)
    
    # Pairwise similarity matrix
    cos_sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

    # Find the 'Centroid' (the candidate most similar to all others)
    consensus_scores = np.mean(cos_sim_matrix, axis=1)
    best_idx = np.argmax(consensus_scores)

    return {
        "final_consensus": candidates[best_idx],
        "confidence": consensus_scores[best_idx],
        "all_paths": candidates
    }

# =====================================================
# EXECUTION
# =====================================================
legal_query = "What are the grounds for an appeal in a Section 302 IPC case?"
result = generate_falcon_consistency(legal_query, num_samples=5)

print("\n" + "="*50)
print("FALCON CONSENSUS SUMMARY")
print("="*50)
print(result['final_consensus'])
print(f"\nConsensus Confidence: {result['confidence']:.4f}")