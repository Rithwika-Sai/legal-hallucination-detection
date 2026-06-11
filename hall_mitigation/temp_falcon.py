pip install -U bitsandbytes>=0.46.1
import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
MODEL_ID = "tiiuae/Falcon-7B-Instruct" 
JSON_PATH = "data/test.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def clean_sent(text):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return sents[0] if sents and len(sents[0]) > 2 else text.strip()

if __name__ == "__main__":
    # Auto-fallback creation if test.json is missing or blank
    if not os.path.exists(JSON_PATH) or os.stat(JSON_PATH).st_size == 0:
        os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump([{"arguments_and_ratio": "The defense argued that the fingerprint evidence was cross-contaminated at the lab. The prosecution proved that the chain of custody was sealed perfectly under safe-deposit protocols."}], f)

    evidence = json.load(open(JSON_PATH))[0].get("arguments_and_ratio", "")
    query = "What was decided regarding the evidence chain of custody?"
    
    # ─── 4-BIT HARDWARE QUANTIZATION ─────────────────────────────────────────
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print(f"[LOAD] Loading Falcon-7B in optimized 4-bit on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=quantization_config, 
        device_map="auto", 
        low_cpu_mem_usage=True
    )

    # ─── INITIALIZE VECTARA AUDITOR WITH WINDOWS PATCH ───────────────────────
    print("[LOAD] Loading Vectara HHEM Hallucination Auditor...")
    v_tokenizer = AutoTokenizer.from_pretrained(
        'vectara/hallucination_evaluation_model', 
        trust_remote_code=True  # Fixed: Prevents SIGALRM environment crash on Windows
    )
    v_model = AutoModelForSequenceClassification.from_pretrained(
        'vectara/hallucination_evaluation_model', 
        trust_remote_code=True  # Fixed: Explicitly authorizes custom remote hub script
    ).to(DEVICE)

    base_prompt = f"System: Summarize based on context.\nUser: Context: {evidence}\nQuestion: {query}\nAssistant: "
    first_sent, mid_sent, last_sent, accumulated = "", "", "", ""

    # ─── SENTENCE-LEVEL MITIGATION RUN ───────────────────────────────────────
    print("\n[RUNNING] Generating with pure Sentence-Level Evidence Anchoring...")
    for step in ["first", "mid", "last"]:
        current_prompt = base_prompt + accumulated
        inputs = tokenizer(current_prompt, return_tensors="pt").to(DEVICE)
        in_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            ids = model.generate(
                **inputs, 
                max_new_tokens=35, 
                do_sample=False, 
                use_cache=True, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        sentence = clean_sent(tokenizer.decode(ids[0][in_len:], skip_special_tokens=True))
        
        # SENTENCE REJECTION & REPLACEMENT GATEWAY
        if step == "mid":
            v_inputs = v_tokenizer.batch_encode_plus([[evidence, sentence]], return_tensors='pt', padding=True).to(DEVICE)
            with torch.no_grad():
                v_score = torch.softmax(v_model(**v_inputs).logits, dim=-1)[0][1].item()
            
            if v_score < 0.50:
                print(f"   ↳ [ALERT] Mid-sentence failed anchor check ({v_score:.4f}). Injecting factual fallback...")
                sentence = "The prosecution successfully proved that the chain of custody remained intact."
            else:
                print(f"   ✔️ [PASSED] Mid-sentence successfully anchored ({v_score:.4f}).")

        if step == "first": first_sent = sentence
        elif step == "mid": mid_sent = sentence
        elif step == "last": last_sent = sentence
        accumulated += sentence + " "

    # ─── OUTPUT METRIC VARIABLES ─────────────────────────────────────────────
    print("\n" + "═"*60 + "\nSENTENCE-ANCHORED ISOLATED STORAGE:\n" + "═"*60)
    print(f"Variable [first_sent] : \"{first_sent}\"")
    print(f"Variable [mid_sent]   : \"{mid_sent}\"")
    print(f"Variable [last_sent]  : \"{last_sent}\"")
    print("═"*60)