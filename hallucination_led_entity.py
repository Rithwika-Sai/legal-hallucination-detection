import os
import warnings
import spacy
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rapidfuzz import process, fuzz

# =====================================================
# 1. SETUP
# =====================================================
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

nlp = spacy.load("en_core_web_sm")
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
embed_model = SentenceTransformer('law-ai/InLegalBERT')

# =====================================================
# 2. EVALUATION ENGINE (ULTRA-STRICT)
# =====================================================
def get_legal_entities(text):
    doc = nlp(text)
    target_labels = {"ORG", "PERSON", "GPE", "LAW", "DATE", "CARDINAL"}
    return set([ent.text.strip() for ent in doc.ents if ent.label_ in target_labels])

def calculate_ultra_strict_hallucination(source_text, summary_text):
    # --- A. SENTENCE-LEVEL LOGIC CHECK ---
    summary_doc = nlp(summary_text)
    summary_sentences = [sent.text.strip() for sent in summary_doc.sents if len(sent.text) > 10]
    
    # Pre-extract source context
    source_nouns = set([t.text.lower() for t in nlp(source_text[:4000]) if t.pos_ == "NOUN"])
    
    sentence_risks = []
    
    for sent in summary_sentences:
        # 1. Contradiction Prob
        nli_logits = nli_model.predict([(source_text[:3000], sent)])
        probs = torch.softmax(torch.tensor(nli_logits), dim=1).tolist()[0]
        
        # Penalize EVERYTHING that isn't Entailment (Label 1)
        # If it's Neutral (Label 2) or Contradiction (Label 0), we treat it as a risk
        non_entailment_risk = (probs[0] + probs[2]) * 100
        
        # 2. Strict Noun Intrusion (The Booster)
        sent_nouns = set([t.text.lower() for t in nlp(sent) if t.pos_ == "NOUN"])
        new_nouns = sent_nouns - source_nouns
        # Heavy penalty: Every new noun is an unverified "fact"
        noun_penalty = len(new_nouns) * 25 
        
        sentence_risks.append((0.6 * non_entailment_risk) + (0.4 * noun_penalty))

    # --- B. ZERO-TOLERANCE ENTITY MATCHING ---
    source_ents = list(get_legal_entities(source_text[:50000]))
    summary_ents = list(get_legal_entities(summary_text))
    
    unsupported_ents = []
    for s_ent in summary_ents:
        # fuzz.ratio is the strictest: "Supreme Court" vs "The Court" will FAIL.
        match = process.extractOne(s_ent, source_ents, scorer=fuzz.ratio)
        if not match or match[1] < 98: # 98% is practically an exact match
            unsupported_ents.append(s_ent)
    
    entity_hallucination_rate = (len(unsupported_ents) / len(summary_ents) * 100) if summary_ents else 0

    # --- C. FINAL AGGREGATION ---
    avg_sentence_risk = sum(sentence_risks) / len(sentence_risks) if sentence_risks else 0
    
    # Final weighting
    final_score = (0.7 * avg_sentence_risk) + (0.3 * entity_hallucination_rate)
    
    return {
        "score": round(min(final_score, 100), 2),
        "flagged": unsupported_ents
    }

# =====================================================
# 3. EXECUTION LOOP
# =====================================================
with open("data/test.json", "r", encoding="utf-8") as f:
    source_content = f.read()

models = {
    "LED": "outputs/led_output.txt"
}

results_list = []
print(f"\n{'MODEL':<15} | {'ULTRA-STRICT HALLUCINATION SCORE':<35} | {'STATUS'}")
print("-" * 80)

for name, path in models.items():
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            summary = f.read().replace("\\n", " ")
        
        res = calculate_ultra_strict_hallucination(source_content, summary)
        status = "❌ FAIL" if res['score'] > 20 else "✅ PASS"
        
        print(f"{name:<15} | {res['score']:>32}% | {status}")
        results_list.append({"Model": name, "Score": res['score'], "Flagged": res['flagged']})
    else:
        print(f"{name:<15} | File Missing")

# Export
pd.DataFrame(results_list).to_csv("Ultra_Strict_Report.csv", index=False)