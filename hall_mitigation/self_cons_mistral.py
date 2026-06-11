import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np

# 1. SETUP MODELS
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

# Semantic model to compare outputs
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def generate_self_consistency(prompt, num_samples=5):
    """
    Generates multiple outputs and selects the one most consistent 
    with the overall group.
    """
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    print(f"Generating {num_samples} candidate paths...")
    
    # --- STEP 1: GENERATE MULTIPLE PATHS ---
    # We enable do_sample and a temperature > 0 to get diverse variations
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode all generated sequences
    candidates = []
    for i in range(num_samples):
        gen_text = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        candidates.append(gen_text)

    # --- STEP 2: CALCULATE CONSENSUS ---
    # Encode candidates into vectors
    embeddings = embedder.encode(candidates, convert_to_tensor=True)
    
    # Compute cosine similarity matrix between all candidates
    cos_sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

    # Calculate the average similarity of each candidate to all others
    # The candidate with the highest average similarity is the "Consensus"
    average_similarities = np.mean(cos_sim_matrix, axis=1)
    best_index = np.argmax(average_similarities)

    return {
        "consensus_output": candidates[best_index],
        "confidence_score": average_similarities[best_index],
        "all_candidates": candidates
    }

# =====================================================
# EXECUTION
# =====================================================
user_prompt = "Summarize the legal implications of Section 144 in this case."
result = generate_self_consistency(user_prompt, num_samples=5)

print("\n" + "="*50)
print("FINAL CONSENSUS OUTPUT")
print("="*50)
print(result['consensus_output'])
print(f"\nConsistency Confidence Score: {result['confidence_score']:.4f}")