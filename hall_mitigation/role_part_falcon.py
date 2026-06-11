import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
# Using Falcon's Instruct checkpoint optimized for alignment
MODEL_ID = "tiiuae/Falcon-7B-Instruct" 
JSON_PATH = "data/test.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── DATA SEPARATION ENGINE ──────────────────────────────────────────────────
def load_role_partitioned_data(json_path):
    """Safely loads and breaks down your legal data structure."""
    if not os.path.exists(json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        sample_data = [{
            "facts": "The appellant was convicted under Section 302 IPC for murder.",
            "arguments": "Defense counsel argues that the eyewitness testimonies are highly contradictory and unreliable. Main prosecution witnesses turned hostile.",
            "ratio_decidendi": "The court held that minor contradictions do not erode the case if circumstantial evidence is solid.",
            "relief": "The criminal appeal is hereby dismissed."
        }]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=4)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data if isinstance(data, list) else [data]

# ─── MAIN OPTIMIZED PIPELINE ────────────────────────────────────────────────
if __name__ == "__main__":
    print("[SPEED OPTIMIZATION] Reading structural data segments...")
    case_partitions = load_role_partitioned_data(JSON_PATH)
    active_case = case_partitions[0]
    
    # Isolate the exact middle section (Arguments) to keep the text window short
    isolated_context = active_case.get("arguments", "Context missing.")
    target_query = "What did the defense counsel argue regarding the witnesses?"
    
    # Enforce strict zero-trust constraints via Falcon-specific prompt styling
    SYSTEM_GUARDRAIL = (
        "You are a legal reader. Answer using ONLY the explicit facts from the "
        "provided Role Context. If the answer is missing, respond with 'Information not found.'"
    )
    
    # Falcon standard format: System -> User -> Assistant
    final_prompt = (
        f"System: {SYSTEM_GUARDRAIL}\n"
        f"User: Role Context: {isolated_context}\nQuestion: {target_query}\n"
        f"Assistant:"
    )
    
    # ─── HIGH-SPEED DISK LOAD ───────────────────────────────────────────────
    print(f"[SPEED OPTIMIZATION] Speed loading Falcon weights onto {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Dynamic matrix data configurations to push hardware limits
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
        # Natively triggers ultra-fast Scaled Dot-Product Attention
        attn_implementation="sdpa" if DEVICE == "cuda" else "eager"
    )
    
    # ─── ULTRA-SPEED INFERENCE ──────────────────────────────────────────────
    print("\n[INFERENCE] Executing ultra-speed token generation...")
    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,           # Low variance prevents logical wander
            use_cache=True,            # Fast KV cache lookup strategy
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    print("\n" + "═"*60)
    print("FALCON HIGH-SPEED PIPELINE SUMMARY")
    print("═"*60)
    print(f"Isolated Segment Context Size: {len(isolated_context.split())} words")
    print(f"Falcon Core Response:\n{generated_text}")
    print("═"*60)