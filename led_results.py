
# =========================================================
# LED LEGAL SUMMARIZATION
# + ROUGE-L
# + BERTScore
# + Ensemble
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install torch
# pip install rouge-score
# pip install bert-score
# pip install pandas
# pip install sentencepiece

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
# =========================================================

source_document = source_document[:1000]

# =========================================================
# REFERENCE SUMMARY
# =========================================================

reference_summary = """
The court reviewed the legal dispute,
examined the evidence and legal provisions,
and delivered the final judgment.
"""

# =========================================================
# LOAD LED MODEL
# =========================================================

print("\nLoading LED Model...\n")

led_model = pipeline(

    "summarization",

    model="allenai/led-base-16384"
)

# =========================================================
# GENERATE SUMMARY
# =========================================================

print("\nGenerating Summary...\n")

summary = led_model(

    source_document,

    max_length=80,

    min_length=20,

    truncation=True,

    do_sample=False

)[0]["summary_text"]

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

    "Model": ["LED"],

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

    "led_results.csv",

    index=False
)

print("\nSaved Successfully:")
print("led_results.csv")

print("\nDONE SUCCESSFULLY\n")
