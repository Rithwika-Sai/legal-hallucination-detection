# import os
# import json
# import re
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer, util

# # ─── CONFIGURATION ──────────────────────────────────────────────────────────
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2" 
# JSON_PATH = "data/test.json"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ─── DATA ENGINE ────────────────────────────────────────────────────────────
# def load_case_partitions(json_path):
#     if not os.path.exists(json_path):
#         os.makedirs(os.path.dirname(json_path), exist_ok=True)
#         sample_data = [{
#             "facts": "The appellant filed an appeal against the High Court order dated Jan 12, 2024.",
#             "arguments_and_ratio": "The defense argued that the fingerprint evidence was cross-contaminated at the lab. The prosecution proved that the chain of custody was sealed perfectly under safe-deposit protocols.",
#             "relief": "The appeal stands dismissed."
#         }]
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(sample_data, f, indent=4)
#     with open(json_path, "r", encoding="utf-8") as f:
#         return json.load(f)[0]

# # ─── HELPER FOR SENTENCE CLEANING ───────────────────────────────────────────
# def extract_first_complete_sentence(text):
#     """Surgically extracts the first coherent sentence out of a token block."""
#     sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#     if sentences and len(sentences[0]) > 2:
#         return sentences[0]
#     return text.strip()

# # ─── MAIN EXECUTION PIPELINE ────────────────────────────────────────────────
# if __name__ == "__main__":
#     case_data = load_case_partitions(JSON_PATH)
    
#     # Target ONLY the mid-level text structure (Arguments & Ratio) where drift occurs
#     mid_level_evidence = case_data.get("arguments_and_ratio", "")
#     user_query = "What was decided regarding the fingerprint evidence chain of custody?"
    
#     # ─── HIGH-SPEED MODELS LOADING ──────────────────────────────────────────
#     print(f"[LOAD] Initializing high-speed text checking models on {DEVICE}...")
#     # Ultra-fast embedder (~2ms per lookup)
#     similarity_checker = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#     tokenizer.pad_token = tokenizer.eos_token
    
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
#         device_map="auto" if DEVICE == "cuda" else None,
#         low_cpu_mem_usage=True,
#         attn_implementation="sdpa" if DEVICE == "cuda" else "eager"
#     )

#     # ─── ANCHORED ITERATIVE GENERATION LOOP ──────────────────────────────────
#     print("\n[LOOP] Starting Sentence-Level Anchoring over mid-level structure...")
    
#     SYSTEM_GUARDRAIL = "You are a legal summarizer. Write a factual sentence answering the question based strictly on the context."
#     base_prompt = f"System: {SYSTEM_GUARDRAIL}\nUser: Context: {mid_level_evidence}\nQuestion: {user_query}\nAssistant: "
    
#     # Pre-encode our target legal context for lightning fast comparative loops
#     evidence_embedding = similarity_checker.encode(mid_level_evidence, convert_to_tensor=True)
    
#     accumulated_summary = ""
#     max_sentences_to_generate = 3
    
#     for i in range(max_sentences_to_generate):
#         # Dynamically append previous true steps into current generation history window
#         current_step_prompt = base_prompt + accumulated_summary
#         inputs = tokenizer(current_step_prompt, return_tensors="pt").to(model.device)
#         input_len = inputs["input_ids"].shape[1]
        
#         # Step 1: Write just ONE sentence (small max_new_tokens ceiling)
#         with torch.no_grad():
#             output_ids = model.generate(
#                 **inputs,
#                 max_new_tokens=40,   # Limits processing window to roughly 1 sentence
#                 do_sample=True,
#                 temperature=0.4,     # Standard operational settings
#                 use_cache=True,      # ULTRA-SPEED: Reuses KV calculations of past steps
#                 pad_token_id=tokenizer.eos_token_id
#             )
        
#         raw_sentence = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
#         clean_sentence = extract_first_complete_sentence(raw_sentence)
        
#         # Step 2: Immediate Check via math embeddings
#         sentence_embedding = similarity_checker.encode(clean_sentence, convert_to_tensor=True)
#         similarity_score = float(util.cos_sim(sentence_embedding, evidence_embedding)[0][0])
        
#         # Step 3 & 4: Error Handling Triggers
#         if similarity_score < 0.55:
#             print(f"  ↳ [REJECTED (Score: {similarity_score:.3f})] \"{clean_sentence}\" -> Triggering Step 3 Re-run...")
            
#             # Step 3: Regenerate immediately with ultra-low, completely objective creativity
#             with torch.no_grad():
#                 retry_ids = model.generate(
#                     **inputs,
#                     max_new_tokens=40,
#                     do_sample=True,
#                     temperature=0.05, # Drop temperature instantly to eliminate hallucination vectors
#                     use_cache=True,
#                     pad_token_id=tokenizer.eos_token_id
#                 )
#             raw_sentence = tokenizer.decode(retry_ids[0][input_len:], skip_special_tokens=True)
#             clean_sentence = extract_first_complete_sentence(raw_sentence)
            
#             # Re-evaluate
#             sentence_embedding = similarity_checker.encode(clean_sentence, convert_to_tensor=True)
#             similarity_score = float(util.cos_sim(sentence_embedding, evidence_embedding)[0][0])
            
#             # Step 4: Final absolute safety filter fallback
#             if similarity_score < 0.55:
#                 print(f"  ↳ [REJECTED AGAIN] Triggering Step 4 Core Fallback Injection...")
#                 # Inject a factual baseline direct from data if the model continues hallucinating
#                 clean_sentence = f"Regarding this matter, the source confirms that: {mid_level_evidence}."
        
#         print(f"  ✔️ [ACCEPTED (Score: {similarity_score:.3f})] Added: \"{clean_sentence}\"")
        
#         # Step 5: Append to structural memory block and iterate forward
#         accumulated_summary += clean_sentence + " "

#     # ─── SYSTEM STATUS PRINTOUT ─────────────────────────────────────────────
#     print("\n" + "═"*60)
#     print("FALCON SENTENCE-ANCHORED MID-LEVEL SUMMARY")
#     print("═"*60)
#     print(f"Final Mitigated Summary:\n{accumulated_summary.strip()}")
#     print("═"*60)
    
    
    
    
    
    
# import os
# import json
# import re
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer, util
 
# # ─── CONFIGURATION ──────────────────────────────────────────────────────────
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
# JSON_PATH = "data/test.json"
# DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
 
# # FIX 1: Raised acceptance threshold 0.55 → 0.70
# #         Weak paraphrases that previously slipped through are now rejected
# ACCEPTANCE_THRESHOLD = 0.70
 
# # FIX 2: Max re-run attempts before safe fallback fires
# #         Was: 1 re-run then fallback to raw source injection
# #         Now: 3 capped re-runs, then "Information not found" — never accepts low-scoring output
# MAX_RERUN_ATTEMPTS = 3
 
# # ─── DATA ENGINE ────────────────────────────────────────────────────────────
# def load_case_partitions(json_path):
#     if not os.path.exists(json_path):
#         os.makedirs(os.path.dirname(json_path), exist_ok=True)
#         sample_data = [{
#             "facts": "The appellant filed an appeal against the High Court order dated Jan 12, 2024.",
#             "arguments_and_ratio": (
#                 "The defense argued that the fingerprint evidence was cross-contaminated at the lab. "
#                 "The prosecution proved that the chain of custody was sealed perfectly under safe-deposit protocols."
#             ),
#             "relief": "The appeal stands dismissed."
#         }]
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(sample_data, f, indent=4)
#     with open(json_path, "r", encoding="utf-8") as f:
#         return json.load(f)[0]
 
# # ─── HELPER FOR SENTENCE CLEANING ───────────────────────────────────────────
# def extract_first_complete_sentence(text):
#     """Surgically extracts the first coherent sentence out of a token block."""
#     sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#     if sentences and len(sentences[0]) > 2:
#         return sentences[0]
#     return text.strip()
 
# # ─── MAIN EXECUTION PIPELINE ────────────────────────────────────────────────
# if __name__ == "__main__":
#     case_data          = load_case_partitions(JSON_PATH)
#     mid_level_evidence = case_data.get("arguments_and_ratio", "")
#     user_query         = "What was decided regarding the fingerprint evidence chain of custody?"
 
#     # ─── MODEL LOADING ───────────────────────────────────────────────────────
#     print(f"[LOAD] Initializing high-speed text checking models on {DEVICE}...")
#     similarity_checker = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
 
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#     tokenizer.pad_token = tokenizer.eos_token
 
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         # FIX (warning): torch_dtype → dtype
#         dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
#         device_map="auto" if DEVICE == "cuda" else None,
#         low_cpu_mem_usage=True,
#         attn_implementation="sdpa" if DEVICE == "cuda" else "eager"
#     )
 
#     # ─── ANCHORED ITERATIVE GENERATION LOOP ─────────────────────────────────
#     print(f"\n[LOOP] Starting Sentence-Level Anchoring  "
#           f"(threshold={ACCEPTANCE_THRESHOLD}, max_reruns={MAX_RERUN_ATTEMPTS})...")
 
#     # FIX 3: Added meta-commentary suppression to system guardrail.
#     #         Previous prompt allowed the model to narrate its own reasoning
#     #         (e.g. "This sentence accurately reflects...") which leaked into output.
#     SYSTEM_GUARDRAIL = (
#         "You are a legal summarizer. "
#         "Write a single factual sentence answering the question based strictly on the context. "
#         "Do not explain or justify your answer. "
#         "Do not narrate your own reasoning. "
#         "Output only the factual answer sentence."
#     )
 
#     base_prompt = (
#         f"<s>[INST] {SYSTEM_GUARDRAIL}\n"
#         f"Context: {mid_level_evidence}\n"
#         f"Question: {user_query} [/INST]"
#     )
 
#     # Pre-encode source context for fast cosine comparison
#     evidence_embedding     = similarity_checker.encode(mid_level_evidence, convert_to_tensor=True)
#     accumulated_summary    = ""
#     max_sentences_to_generate = 3
 
#     for i in range(max_sentences_to_generate):
#         current_step_prompt = base_prompt + accumulated_summary
#         inputs    = tokenizer(current_step_prompt, return_tensors="pt").to(model.device)
#         input_len = inputs["input_ids"].shape[1]
 
#         # ── Step 1: Generate one sentence ────────────────────────────────────
#         with torch.no_grad():
#             output_ids = model.generate(
#                 **inputs,
#                 max_new_tokens=40,
#                 do_sample=True,
#                 temperature=0.4,
#                 use_cache=True,
#                 pad_token_id=tokenizer.eos_token_id
#             )
 
#         raw_sentence   = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
#         clean_sentence = extract_first_complete_sentence(raw_sentence)
 
#         # ── Step 2: Similarity check against source ───────────────────────────
#         sent_emb        = similarity_checker.encode(clean_sentence, convert_to_tensor=True)
#         similarity_score = float(util.cos_sim(sent_emb, evidence_embedding)[0][0])
 
#         # ── FIX 1 + FIX 2: Threshold 0.70, capped re-run loop ────────────────
#         if similarity_score < ACCEPTANCE_THRESHOLD:
#             print(f"  ✘ [REJECTED  score={similarity_score:.3f}] \"{clean_sentence}\" "
#                   f"→ Triggering re-run (max {MAX_RERUN_ATTEMPTS})...")
 
#             accepted = False
#             for attempt in range(1, MAX_RERUN_ATTEMPTS + 1):
#                 with torch.no_grad():
#                     retry_ids = model.generate(
#                         **inputs,
#                         max_new_tokens=40,
#                         do_sample=True,
#                         # Drop temperature each attempt for tighter factual lock
#                         temperature=max(0.4 - attempt * 0.1, 0.05),
#                         use_cache=True,
#                         pad_token_id=tokenizer.eos_token_id
#                     )
 
#                 raw_sentence   = tokenizer.decode(retry_ids[0][input_len:], skip_special_tokens=True)
#                 clean_sentence = extract_first_complete_sentence(raw_sentence)
#                 sent_emb       = similarity_checker.encode(clean_sentence, convert_to_tensor=True)
#                 similarity_score = float(util.cos_sim(sent_emb, evidence_embedding)[0][0])
 
#                 if similarity_score >= ACCEPTANCE_THRESHOLD:
#                     print(f"    ✔ [RE-RUN {attempt} ACCEPTED  score={similarity_score:.3f}] "
#                           f"\"{clean_sentence}\"")
#                     accepted = True
#                     break
#                 else:
#                     print(f"    ✘ [RE-RUN {attempt} REJECTED  score={similarity_score:.3f}] "
#                           f"\"{clean_sentence}\"")
 
#             # FIX 2: All re-runs exhausted → safe fallback, never inject raw source
#             if not accepted:
#                 clean_sentence   = "Information not found within this role."
#                 similarity_score = 0.0
#                 print(f"    ⚠ All {MAX_RERUN_ATTEMPTS} re-runs failed. "
#                       f"Falling back: \"{clean_sentence}\"")
 
#         else:
#             print(f"  ✔ [ACCEPTED  score={similarity_score:.3f}] \"{clean_sentence}\"")
 
#         # ── Step 5: Append accepted sentence and iterate ──────────────────────
#         accumulated_summary += clean_sentence + " "
 
#     # ─── OUTPUT ──────────────────────────────────────────────────────────────
#     print("\n" + "═"*60)
#     print("MISTRAL SENTENCE-ANCHORED MID-LEVEL SUMMARY")
#     print("═"*60)
#     print(f"Final Mitigated Summary:\n{accumulated_summary.strip()}")
#     print("═"*60)
    
    
    
    
    
    
    
import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
 
# ─── CONFIGURATION ──────────────────────────────────────────────────────────
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
JSON_PATH = "data/test.json"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
 
ACCEPTANCE_THRESHOLD = 0.70   # FIX 1: was 0.55
MAX_RERUN_ATTEMPTS   = 3      # FIX 2: was 1 re-run then raw source injection
 
# ─── DATA ENGINE ────────────────────────────────────────────────────────────
def load_case_partitions(json_path):
    if not os.path.exists(json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        sample_data = [{
            "facts": "The appellant filed an appeal against the High Court order dated Jan 12, 2024.",
            "arguments_and_ratio": (
                "The defense argued that the fingerprint evidence was cross-contaminated at the lab. "
                "The prosecution proved that the chain of custody was sealed perfectly under safe-deposit protocols."
            ),
            "relief": "The appeal stands dismissed."
        }]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=4)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)[0]
 
# ─── SENTENCE CLEANER ───────────────────────────────────────────────────────
def extract_first_complete_sentence(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if sentences and len(sentences[0]) > 2:
        return sentences[0]
    return text.strip()
 
# ─── GENERATION HELPER ──────────────────────────────────────────────────────
def generate_one_sentence(model, tokenizer, inputs, input_len,
                          similarity_checker, evidence_embedding, temp):
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=temp,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    raw   = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    clean = extract_first_complete_sentence(raw)
    emb   = similarity_checker.encode(clean, convert_to_tensor=True)
    score = float(util.cos_sim(emb, evidence_embedding)[0][0])
    return clean, score
 
# ─── MAIN PIPELINE ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    case_data          = load_case_partitions(JSON_PATH)
    mid_level_evidence = case_data.get("arguments_and_ratio", "")
    user_query         = "What was decided regarding the fingerprint evidence chain of custody?"
 
    print(f"[LOAD] Initializing high-speed text checking models on {DEVICE}...")
    similarity_checker = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
 
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if DEVICE == "cuda" else "eager",
        trust_remote_code=True   # Required for Falcon custom architecture
    )
 
    print(f"\n[LOOP] Starting Sentence-Level Anchoring "
          f"(threshold={ACCEPTANCE_THRESHOLD}, max_reruns={MAX_RERUN_ATTEMPTS})...")
 
    # FIX 3: Meta-commentary suppression — stops model narrating its own reasoning
    SYSTEM_GUARDRAIL = (
        "You are a legal summarizer. "
        "Write a single factual sentence answering the question based strictly on the context. "
        "Do not explain or justify your answer. "
        "Do not narrate your own reasoning. "
        "Output only the factual answer sentence."
    )
 
    # Falcon instruction format: System / User / Assistant
    base_prompt = (
        f"System: {SYSTEM_GUARDRAIL}\n"
        f"User: Context: {mid_level_evidence}\n"
        f"Question: {user_query}\n"
        f"Assistant:"
    )
 
    evidence_embedding        = similarity_checker.encode(mid_level_evidence, convert_to_tensor=True)
    accumulated_summary       = ""
    max_sentences_to_generate = 3
 
    for i in range(max_sentences_to_generate):
        inputs    = tokenizer(base_prompt + accumulated_summary, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
 
        clean_sentence, similarity_score = generate_one_sentence(
            model, tokenizer, inputs, input_len,
            similarity_checker, evidence_embedding, temp=0.4
        )
 
        if similarity_score < ACCEPTANCE_THRESHOLD:
            print(f"  ✘ [REJECTED  score={similarity_score:.3f}] \"{clean_sentence}\" "
                  f"→ Triggering re-run (max {MAX_RERUN_ATTEMPTS})...")
            accepted = False
            for attempt in range(1, MAX_RERUN_ATTEMPTS + 1):
                retry_temp = max(0.4 - attempt * 0.1, 0.05)
                clean_sentence, similarity_score = generate_one_sentence(
                    model, tokenizer, inputs, input_len,
                    similarity_checker, evidence_embedding, temp=retry_temp
                )
                if similarity_score >= ACCEPTANCE_THRESHOLD:
                    print(f"    ✔ [RE-RUN {attempt} ACCEPTED  score={similarity_score:.3f}] "
                          f"\"{clean_sentence}\"")
                    accepted = True
                    break
                else:
                    print(f"    ✘ [RE-RUN {attempt} REJECTED  score={similarity_score:.3f}] "
                          f"\"{clean_sentence}\"")
 
            # FIX 2: clean fallback — never injects raw source text
            if not accepted:
                clean_sentence   = "Information not found within this role."
                similarity_score = 0.0
                print(f"    ⚠ All {MAX_RERUN_ATTEMPTS} re-runs failed. "
                      f"Falling back: \"{clean_sentence}\"")
        else:
            print(f"  ✔ [ACCEPTED  score={similarity_score:.3f}] \"{clean_sentence}\"")
 
        accumulated_summary += clean_sentence + " "
 
    print("\n" + "═"*60)
    print("FALCON SENTENCE-ANCHORED MID-LEVEL SUMMARY")
    print("═"*60)
    print(f"Final Mitigated Summary:\n{accumulated_summary.strip()}")
    print("═"*60)