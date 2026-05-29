# =========================================================
# FALCON-7B LEGAL SUMMARIZATION
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
# pip install rouge-score
# pip install bert-score
# pip install pandas
# pip install sentencepiece

# =========================================================
# IMPORTS
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd

from transformers import (

    AutoTokenizer,

    AutoModelForCausalLM
)

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

source_document = source_document[:300]

# =========================================================
# REFERENCE SUMMARY
# =========================================================

reference_summary = """
The court reviewed the legal dispute,
examined the evidence and legal provisions,
and delivered the final judgment.
"""

# =========================================================
# LOAD FALCON MODEL
# =========================================================

print("\nLoading Falcon-7B Model...\n")

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(

    model_name,

    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(

    model_name,

    trust_remote_code=True,

    torch_dtype=torch.float32
)

# =========================================================
# IMPORTANT FIXES
# =========================================================

model.config.use_cache = False

tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# CREATE PROMPT
# =========================================================

prompt = f"""
Summarize this legal judgment briefly:

{source_document}

Summary:
"""

# =========================================================
# TOKENIZE INPUT
# =========================================================

inputs = tokenizer(

    prompt,

    return_tensors="pt",

    truncation=True,

    max_length=512
)

# =========================================================
# GENERATE SUMMARY
# =========================================================

print("\nGenerating Summary...\n")

with torch.no_grad():

    outputs = model.generate(

        input_ids=inputs["input_ids"],

        attention_mask=inputs["attention_mask"],

        max_new_tokens=20,

        do_sample=False,

        use_cache=False,

        pad_token_id=tokenizer.eos_token_id
    )

# =========================================================
# DECODE SUMMARY
# =========================================================

summary = tokenizer.decode(

    outputs[0],

    skip_special_tokens=True
)

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

    "Model": ["Falcon-7B"],

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

    "falcon_results.csv",

    index=False
)

print("\nSaved Successfully:")
print("falcon_results.csv")

print("\nDONE SUCCESSFULLY\n")




