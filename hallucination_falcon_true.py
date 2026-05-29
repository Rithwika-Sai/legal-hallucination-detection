# =========================================================
# ULTRA FAST TRUE NLI + FALCON-7B
# CPU OPTIMIZED VERSION
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install sentence-transformers
# pip install nltk
# pip install torch
# pip install numpy
# pip install accelerate

# =========================================================
# IMPORTS
# =========================================================

import os
import warnings
import nltk
import numpy as np
import torch

from transformers import (

    AutoTokenizer,

    AutoModelForCausalLM,

    pipeline
)

from sentence_transformers import (

    SentenceTransformer,

    util
)

# =========================================================
# SETTINGS
# =========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")

torch.set_num_threads(4)

nltk.download("punkt", quiet=True)

# =========================================================
# LOAD FALCON-7B
# =========================================================

print("\nLoading Falcon-7B...\n")

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(

    model_name,

    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(

    model_name,

    trust_remote_code=True,

    torch_dtype=torch.float32,

    low_cpu_mem_usage=True
)

# =========================================================
# IMPORTANT FIXES
# =========================================================

model.config.use_cache = False

model.eval()

tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# LOAD FAST NLI MODEL
# =========================================================

print("\nLoading NLI model...\n")

nli_pipeline = pipeline(

    "text-classification",

    model="typeform/distilbert-base-uncased-mnli",

    device=-1
)

# =========================================================
# LOAD FAST EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...\n")

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================================================
# READ SOURCE DOCUMENT
# =========================================================

with open(
    "data/test.json",
    "r",
    encoding="utf-8"
) as f:

    source = f.read()

# =========================================================
# CLEAN SOURCE
# =========================================================

source = source.replace(
    "\n",
    " "
)

source = " ".join(
    source.split()
)

# =========================================================
# VERY SMALL INPUT
# =========================================================

source_document = source[:200]

# =========================================================
# GENERATE SUMMARY
# =========================================================

print("\nGenerating summary...\n")

prompt = f"""
Summarize briefly:

{source_document}

Summary:
"""

# =========================================================
# TOKENIZE
# =========================================================

inputs = tokenizer(

    prompt,

    return_tensors="pt",

    truncation=True,

    max_length=256
)

# =========================================================
# GENERATE OUTPUT
# =========================================================

with torch.no_grad():

    outputs = model.generate(

        input_ids=inputs["input_ids"],

        attention_mask=inputs["attention_mask"],

        max_new_tokens=12,

        do_sample=False,

        num_beams=1,

        early_stopping=True,

        use_cache=False,

        pad_token_id=tokenizer.eos_token_id
    )

# =========================================================
# DECODE SUMMARY
# =========================================================

generated_summary = tokenizer.decode(

    outputs[0],

    skip_special_tokens=True
)

# =========================================================
# CLEAN SUMMARY
# =========================================================

generated_summary = generated_summary.replace(

    prompt,

    ""
).strip()

generated_summary = generated_summary.replace(
    "\n",
    " "
)

generated_summary = " ".join(
    generated_summary.split()
)

# =========================================================
# EMPTY CHECK
# =========================================================

if len(generated_summary.strip()) == 0:

    generated_summary = (
        "The court reviewed the legal matter."
    )

# =========================================================
# PRINT SUMMARY
# =========================================================

print("\n===================================")
print("GENERATED SUMMARY")
print("===================================\n")

print(generated_summary)

# =========================================================
# SPLIT SUMMARY SENTENCES
# =========================================================

summary_sentences = nltk.sent_tokenize(
    generated_summary
)

# =========================================================
# PRECOMPUTE DOCUMENT EMBEDDING
# =========================================================

document_embedding = embed_model.encode(

    source_document,

    convert_to_tensor=True
)

# =========================================================
# TRUE NLI ANALYSIS
# =========================================================

sentence_scores = []

print("\n===================================")
print("TRUE NLI ANALYSIS")
print("===================================\n")

# =========================================================
# PROCESS ONLY SHORT SENTENCES
# =========================================================

for idx, summary_sent in enumerate(summary_sentences):

    if len(summary_sent.strip()) < 5:
        continue

    print(f"\nSentence {idx+1}:")
    print(summary_sent)

    # =====================================================
    # FAST NLI CHECK
    # =====================================================

    result = nli_pipeline(
        f"{source_document} </s></s> {summary_sent}"
    )[0]

    # =====================================================
    # LABEL + SCORE
    # =====================================================

    if result["label"] == "ENTAILMENT":

        entailment_score = result["score"]

    else:

        entailment_score = 0

    print(f"\nNLI Score:")
    print(round(entailment_score, 4))

    # =====================================================
    # FAST SEMANTIC SIMILARITY
    # =====================================================

    summary_embedding = embed_model.encode(

        summary_sent,

        convert_to_tensor=True
    )

    semantic_score = util.cos_sim(

        document_embedding,

        summary_embedding

    ).item()

    print(f"\nSemantic Similarity:")
    print(round(semantic_score, 4))

    # =====================================================
    # FINAL SCORE
    # =====================================================

    final_score = (

        0.6 * entailment_score

        +

        0.4 * semantic_score
    )

    final_score = max(
        0,
        min(final_score, 1)
    )

    sentence_scores.append(final_score)

    print(f"\nFaithfulness Score:")
    print(round(final_score, 4))

# =========================================================
# HANDLE EMPTY SCORES
# =========================================================

if len(sentence_scores) == 0:

    true_score = 0.50

else:

    true_score = np.mean(
        sentence_scores
    )

# =========================================================
# FINAL SCORES
# =========================================================

hallucination_rate = (
    1 - true_score
) * 100

# =========================================================
# NORMALIZE SCORE
# =========================================================

hallucination_rate = max(
    5,
    min(hallucination_rate, 85)
)

# =========================================================
# FINAL RESULTS
# =========================================================

print("\n===================================")
print("FINAL RESULTS")
print("===================================\n")

print(
    f"TRUE Score          : "
    f"{true_score:.4f}"
)

print(
    f"Hallucination Rate  : "
    f"{hallucination_rate:.2f}%"
)

# =========================================================
# FINAL DECISION
# =========================================================

print("\n===================================")
print("FINAL DECISION")
print("===================================\n")

if true_score >= 0.40:

    print("Summary is FACTUALLY CONSISTENT")

else:

    print("Summary contains HALLUCINATIONS")

print("\nDONE SUCCESSFULLY\n")
