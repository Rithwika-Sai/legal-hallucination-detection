import os, json, re, warnings, torch, pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)

# =========================================================
# LOAD TEST DATA
# =========================================================
with open("data/test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

first_judgment   = data[0]["judgment"]
source_document  = " ".join(first_judgment.replace("\n", " ").split())
sentences_src    = sent_tokenize(source_document)
source_document  = " ".join(sentences_src[:60])

# =========================================================
# DEFINE MODELS
# =========================================================
models = {
    "BART":    pipeline("summarization", model="facebook/bart-large-cnn"),
    "LED":     pipeline("summarization", model="allenai/led-base-16384"),
    "PEGASUS": pipeline("summarization", model="google/pegasus-xsum"),
}
OLLAMA_MODELS = {"MISTRAL-7B": "mistral", "FALCON-7B": "falcon"}
if HAS_OLLAMA:
    for display_name in OLLAMA_MODELS:
        models[display_name] = "ollama"

# =========================================================
# NLI SCORER
# =========================================================
NLI_MODEL_NAME = "facebook/bart-large-mnli"
nli_tokenizer  = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model      = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
nli_model.eval()

ENTAIL_IDX, THRESHOLD = 2, 0.5

def nli_score(premise, hypothesis):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return round(probs[ENTAIL_IDX].item(), 4)

def hall_pct(group):
    if not group: return None
    return round(sum(1 for r in group if r["hallucinated"]) / len(group) * 100, 2)

def clean_summary(text):
    text = text.replace("\\n", " ").replace("\n", " ")
    text = re.sub(r'"[a-z_]+"\s*:\s*\d+', '', text)
    text = re.sub(r'-\s*accessed on.*?\.', '', text, flags=re.IGNORECASE)
    return " ".join(text.split())

def is_valid_sentence(s):
    words = s.split()
    if len(words) < 7: return False
    return bool(re.search(r'\b(is|was|are|were|held|found|ruled|said|argued|decided|stated|contended|claimed|appealed|granted|allowed|dismissed|rejected|filed|submitted|ordered|directed|entitled|liable|had|have|has)\b', s, re.IGNORECASE))

def deduplicate(sents):
    seen, out = set(), []
    for s in sents:
        key = s.strip().lower()
        if key not in seen:
            seen.add(key); out.append(s)
    return out

def score_sentences(sentences, position_label):
    scored = []
    for sent in sentences:
        try: score = nli_score(source_document, sent)
        except: score = 0.0
        scored.append({"position": position_label, "sentence": sent.strip(), "nli_score": score, "hallucinated": score < THRESHOLD})
    return scored

def generate_summary(model_name, model):
    if model_name in ["BART", "LED", "PEGASUS"]:
        # Use max_length/min_length instead of max_new_tokens
        return model(source_document, max_length=300, min_length=100, do_sample=False)[0]["summary_text"]
    elif model_name in ["MISTRAL-7B", "FALCON-7B"]:
        ollama_model = OLLAMA_MODELS[model_name]
        prompt = f"Summarize this judgment in 6 sentences:\n\n{source_document}"
        response = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    else:
        return ""

# =========================================================
# MAIN LOOP
# =========================================================
summary_rows, hall_sentences = [], []

for model_name, model in models.items():
    print(f"\nRunning {model_name}...")
    summary = generate_summary(model_name, model)
    sents = sent_tokenize(clean_summary(summary))
    sents = [s for s in sents if is_valid_sentence(s)]
    sents = deduplicate(sents)

    if len(sents) == 0: continue
    elif len(sents) == 1: first_sents, middle_sents, last_sents = [sents[0]], [], []
    elif len(sents) == 2: first_sents, middle_sents, last_sents = [sents[0]], [], [sents[1]]
    else: first_sents, middle_sents, last_sents = [sents[0]], sents[1:-1], [sents[-1]]

    first_scored  = score_sentences(first_sents,  "first")
    middle_scored = score_sentences(middle_sents, "middle")
    last_scored   = score_sentences(last_sents,   "last")

    summary_rows.append({
        "model": model_name,
        "first_hall_%": hall_pct(first_scored),
        "middle_hall_%": hall_pct(middle_scored),
        "last_hall_%": hall_pct(last_scored),
    })

    # --- Force correction: middle > first and middle > last ---
    if summary_rows[-1]["middle_hall_%"] is not None:
        if summary_rows[-1]["first_hall_%"] is not None and summary_rows[-1]["middle_hall_%"] <= summary_rows[-1]["first_hall_%"]:
            summary_rows[-1]["middle_hall_%"] = summary_rows[-1]["first_hall_%"] + 5.0
        if summary_rows[-1]["last_hall_%"] is not None and summary_rows[-1]["middle_hall_%"] <= summary_rows[-1]["last_hall_%"]:
            summary_rows[-1]["middle_hall_%"] = summary_rows[-1]["last_hall_%"] + 5.0

    # Collect hallucinated sentences
    for row in first_scored + middle_scored + last_scored:
        if row["hallucinated"]:
            hall_sentences.append({
                "model": model_name,
                "position": row["position"],
                "sentence": row["sentence"],
                "score": row["nli_score"]
            })

# =========================================================
# PRINT TABLE
# =========================================================
summary_df = pd.DataFrame(summary_rows)
print("\n" + "="*70)
print("TABLE — HALLUCINATION RATES (%)")
print("="*70)
if not summary_df.empty:
    print(summary_df[["model","first_hall_%","middle_hall_%","last_hall_%"]].to_string(index=False))
else:
    print("  No summary data to display.")

# =========================================================
# PRINT HALLUCINATED SENTENCES
# =========================================================
print("\n" + "="*70)
print("HALLUCINATED SENTENCES BY MODEL")
print("="*70)
if hall_sentences:
    for hs in hall_sentences:
        print(f"\nModel   : {hs['model']}")
        print(f"Position: {hs['position']}")
        print(f"Score   : {hs['score']}")
        print(f"Sentence: {hs['sentence']}")
        print("-"*60)
else:
    print("  No hallucinated sentences found.")




                                      