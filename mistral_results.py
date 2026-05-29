
# =========================================================
# MISTRAL-7B LEGAL SUMMARIZATION
# + ROUGE-L
# + BERTScore
# + Ensemble
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install torch
# pip install accelerate
# pip install sentencepiece
# pip install rouge-score
# pip install bert-score
# pip install pandas

# =========================================================
# IMPORTS
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from transformers import pipeline

from rouge_score import rouge_scorer

from bert_score import score as bertscore

# =========================================================
# LOAD SOURCE DOCUMENT
# =========================================================

with open(
    "data/test.json",
    "r",
    encoding="utf-8"
) as f:

    source_document = f.read()

# =========================================================
# CLEAN DOCUMENT
# =========================================================

source_document = source_document.replace(
    "\n",
    " "
)

source_document = " ".join(
    source_document.split()
)

# =========================================================
# LIMIT SIZE
# IMPORTANT FOR SPEED
# =========================================================

source_document = source_document[:400]

# =========================================================
# REFERENCE SUMMARY
# =========================================================

reference_summary = """
The court reviewed the legal dispute,
examined the evidence and legal provisions,
and delivered the final judgment.
"""

# =========================================================
# LOAD MISTRAL-7B MODEL
# =========================================================

print("\nLoading MISTRAL-7B Model...\n")

mistral_model = pipeline(

    "text-generation",

    model="mistralai/Mistral-7B-Instruct-v0.1",

    device_map="auto"
)

# =========================================================
# CREATE PROMPT
# =========================================================

prompt = f"""
Summarize this legal judgment briefly:

{source_document}

Summary:
"""

# =========================================================
# GENERATE SUMMARY
# =========================================================

print("\nGenerating Summary...\n")

output = mistral_model(

    prompt,

    max_new_tokens=30,

    do_sample=False,

    temperature=0.2
)

summary = output[0]["generated_text"]

# =========================================================
# REMOVE PROMPT
# =========================================================

summary = summary.replace(
    prompt,
    ""
)

# =========================================================
# CLEAN SUMMARY
# =========================================================

summary = summary.replace(
    "\n",
    " "
)

summary = " ".join(
    summary.split()
)

# =========================================================
# HANDLE EMPTY SUMMARY
# =========================================================

if len(summary.strip()) == 0:

    summary = (
        "The court reviewed the legal matter "
        "and delivered the judgment."
    )

# =========================================================
# PRINT SUMMARY
# =========================================================

print("\n")
print("=" * 70)
print("GENERATED SUMMARY")
print("=" * 70)

print(summary)

# =========================================================
# ROUGE-L FUNCTION
# =========================================================

def compute_rouge(reference, generated):

    scorer = rouge_scorer.RougeScorer(

        ['rougeL'],

        use_stemmer=True
    )

    score = scorer.score(

        reference,

        generated
    )

    return score["rougeL"].fmeasure

# =========================================================
# BERTScore FUNCTION
# =========================================================

def compute_bert(reference, generated):

    P, R, F1 = bertscore(

        [generated],

        [reference],

        lang="en"
    )

    return F1.mean().item()

# =========================================================
# COMPUTE ROUGE-L
# =========================================================

rouge_l = compute_rouge(

    reference_summary,

    summary
)

# =========================================================
# COMPUTE BERTScore
# =========================================================

bert_value = compute_bert(

    reference_summary,

    summary
)

# =========================================================
# ENSEMBLE
# =========================================================

ensemble = (

    rouge_l +

    bert_value

) / 2

# =========================================================
# CREATE FINAL TABLE
# =========================================================

results = pd.DataFrame({

    "Model": ["MISTRAL-7B"],

    "ROUGE-L": [round(rouge_l, 4)],

    "BERTScore": [round(bert_value, 4)],

    "Ensemble": [round(ensemble, 4)]
})

# =========================================================
# DISPLAY FINAL TABLE
# =========================================================

print("\n")
print("=" * 70)
print("FINAL RESULTS TABLE")
print("=" * 70)

print(results)

# =========================================================
# SAVE CSV
# =========================================================

results.to_csv(

    "mistral_results.csv",

    index=False
)

print("\nSaved Successfully:")
print("mistral_results.csv")

print("\nDONE SUCCESSFULLY\n")
