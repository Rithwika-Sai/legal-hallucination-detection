import os
import warnings
import spacy
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rapidfuzz import process, fuzz

# =====================================================
# 1. INITIALIZATION
# =====================================================
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load Models (Ensure these are installed: spacy, sentence-transformers, rapidfuzz)
nlp = spacy.load("en_core_web_sm")
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

# =====================================================
# 2. EVALUATION FUNCTIONS
# =====================================================
def get_legal_entities(text):
    doc = nlp(text)
    # Focusing on high-stakes legal entities
    target_labels = {"ORG", "PERSON", "GPE", "LAW", "DATE", "CARDINAL"}
    return set([ent.text.strip() for ent in doc.ents if ent.label_ in target_labels])

def calculate_ultra_strict_hallucination(source_text, summary_text):
    summary_doc = nlp(summary_text)
    summary_sentences = [sent.text.strip() for sent in summary_doc.sents if len(sent.text) > 10]
    
    # Pre-extract source nouns (Limit source text for speed/memory)
    source_nouns = set([t.text.lower() for t in nlp(source_text[:4000]) if t.pos_ == "NOUN"])
    
    sentence_risks = []
    for sent in summary_sentences:
        # 1. NLI Logic Check (Neutral + Contradiction = Risk)
        nli_logits = nli_model.predict([(source_text[:3000], sent)])
        probs = torch.softmax(torch.tensor(nli_logits), dim=1).tolist()[0]
        non_entailment_risk = (probs[0] + probs[2]) * 100
        
        # 2. Noun Intrusion (New info not in source)
        sent_nouns = set([t.text.lower() for t in nlp(sent) if t.pos_ == "NOUN"])
        new_nouns = sent_nouns - source_nouns
        noun_penalty = len(new_nouns) * 25 # High multiplier for strictness
        
        sentence_risks.append((0.6 * non_entailment_risk) + (0.4 * noun_penalty))

    # 3. Strict Entity Verification
    source_ents = list(get_legal_entities(source_text[:50000]))
    summary_ents = list(get_legal_entities(summary_text))
    
    unsupported_ents = []
    for s_ent in summary_ents:
        # Using fuzz.ratio for exact-match strictness
        match = process.extractOne(s_ent, source_ents, scorer=fuzz.ratio)
        if not match or match[1] < 98: 
            unsupported_ents.append(s_ent)
    
    entity_hallucination_rate = (len(unsupported_ents) / len(summary_ents) * 100) if summary_ents else 0
    avg_sentence_risk = sum(sentence_risks) / len(sentence_risks) if sentence_risks else 0
    
    # Weighting: Logic (70%) + Entities (30%)
    final_score = (0.7 * avg_sentence_risk) + (0.3 * entity_hallucination_rate)
    
    return {"score": round(min(final_score, 100), 2), "flagged": unsupported_ents}

# =====================================================
# 3. MAIN PROCESSING LOOP
# =====================================================
if not os.path.exists("data/test.json"):
    print("❌ Error: data/test.json not found!")
else:
    with open("data/test.json", "r", encoding="utf-8") as f:
        source_content = f.read()

    # Dictionary of models to check
    models = {
        "Mistral-7B": "outputs/mistral_output.txt",
        "Falcon-7B": "outputs/falcon_output.txt"
    }

    results_list = []
    print(f"\n{'MODEL':<15} | {'ULTRA-STRICT HALLUCINATION SCORE':<35} | {'STATUS'}")
    print("-" * 80)

    for name, path in models.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                summary = f.read().replace("\\n", " ")
            
            res = calculate_ultra_strict_hallucination(source_content, summary)
            
            # Threshold for Legal Fail: > 20%
            status = "❌ FAIL" if res['score'] > 20 else "✅ PASS"
            
            print(f"{name:<15} | {res['score']:>32}% | {status}")
            results_list.append({
                "Model": name, 
                "Score": res['score'], 
                "Unsupported_Entities": ", ".join(res['flagged'])
            })
        else:
            print(f"{name:<15} | ⚠️ File Missing: {path}")

    # Save to CSV for your project report
    if results_list:
        pd.DataFrame(results_list).to_csv("Falcon_Mistral_Audit.csv", index=False)
        print("\n[✔] Results saved to 'Falcon_Mistral_Audit.csv'")



