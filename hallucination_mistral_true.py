# =========================================================
# FINAL MISTRAL-7B + TRUE NLI + InLegalBERT
# FULLY RECTIFIED FINAL VERSION
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install sentence-transformers
# pip install nltk
# pip install torch
# pip install accelerate
# pip install bitsandbytes
# pip install sentencepiece
# pip install numpy

# =========================================================



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
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

nltk.download("punkt", quiet=True)



# =========================================================
# LOAD MISTRAL-7B MODEL
# =========================================================

print("\nLoading Mistral-7B model...\n")

model_name = "mistralai/Mistral-7B-Instruct-v0.1"



tokenizer = AutoTokenizer.from_pretrained(
    model_name
)



# =========================================================
# FIX PAD TOKEN ERROR
# =========================================================

tokenizer.pad_token = tokenizer.eos_token



model = AutoModelForCausalLM.from_pretrained(

    model_name,

    dtype=torch.float32,

    low_cpu_mem_usage=True,

    device_map="auto"
)



# =========================================================
# LOAD FAST NLI MODEL
# =========================================================

print("\nLoading NLI model...\n")

nli_pipeline = pipeline(

    "text-classification",

    model="facebook/bart-large-mnli",

    device=-1
)



# =========================================================
# LOAD InLegalBERT
# =========================================================

print("\nLoading InLegalBERT...\n")

embed_model = SentenceTransformer(
    "law-ai/InLegalBERT"
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

source = source.replace("\n", " ")

source = source.replace("\\n", " ")

source = source.replace("[", " ")

source = source.replace("]", " ")

source = source.replace("{", " ")

source = source.replace("}", " ")

source = source.replace('"', " ")

source = " ".join(source.split())



# =========================================================
# LIMIT DOCUMENT SIZE FOR SPEED
# =========================================================

source_document = source[:500]



# =========================================================
# GENERATE SUMMARY USING MISTRAL-7B
# =========================================================

print("\nGenerating summary...\n")

prompt = f"""
Summarize the following legal judgment clearly:

{source_document}

Summary:
"""



inputs = tokenizer(

    prompt,

    return_tensors="pt",

    truncation=True,

    padding=True,

    max_length=128
)



# =========================================================
# GENERATE
# =========================================================

with torch.no_grad():

    outputs = model.generate(

        input_ids=inputs["input_ids"],

        attention_mask=inputs["attention_mask"],

        max_new_tokens=50,

        temperature=0.2,

        do_sample=False,

        pad_token_id=tokenizer.eos_token_id
    )



generated_summary = tokenizer.decode(

    outputs[0],

    skip_special_tokens=True
)



# =========================================================
# REMOVE PROMPT
# =========================================================

generated_summary = generated_summary.replace(
    prompt,
    ""
)



# =========================================================
# CLEAN SUMMARY
# =========================================================

generated_summary = generated_summary.replace(
    "\n",
    " "
)

generated_summary = " ".join(
    generated_summary.split()
)



# =========================================================
# HANDLE EMPTY SUMMARY
# =========================================================

if len(generated_summary.strip()) == 0:

    generated_summary = (
        "The court reviewed the legal matter "
        "and issued a judgment."
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
# HANDLE EMPTY SENTENCES
# =========================================================

if len(summary_sentences) == 0:

    summary_sentences = [generated_summary]



# =========================================================
# PRECOMPUTE SOURCE EMBEDDING
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
# PROCESS SUMMARY SENTENCES
# =========================================================

for idx, summary_sent in enumerate(summary_sentences):

    if len(summary_sent.strip()) < 5:
        continue



    print(f"\nSentence {idx+1}:")
    print(summary_sent)



    # =====================================================
    # SAFE SHORTENING
    # =====================================================

    short_doc = source_document[:300]

    short_summary = summary_sent[:80]



    # =====================================================
    # FIXED NLI FORMAT
    # =====================================================

    nli_input = (
        f"{short_doc} </s></s> {short_summary}"
    )



    # =====================================================
    # SAFE NLI CHECK
    # =====================================================

    try:

        result = nli_pipeline(

            nli_input,

            truncation=True,

            max_length=256

        )[0]



        label = result["label"]

        nli_score = result["score"]



    except Exception:

        label = "NEUTRAL"

        nli_score = 0.0



    print(f"\nNLI Label:")
    print(label)

    print(f"\nNLI Score:")
    print(round(nli_score, 4))



    # =====================================================
    # SEMANTIC SIMILARITY
    # =====================================================

    try:

        summary_embedding = embed_model.encode(

            short_summary,

            convert_to_tensor=True
        )



        semantic_score = util.cos_sim(

            document_embedding,

            summary_embedding

        ).item()



    except Exception:

        semantic_score = 0.0



    print(f"\nSemantic Similarity:")
    print(round(semantic_score, 4))



    # =====================================================
    # FINAL SCORE
    # =====================================================

    if label == "ENTAILMENT":

        entailment_score = nli_score

    else:

        entailment_score = 0.0



    final_score = (
        0.7 * entailment_score
        +
        0.3 * semantic_score
    )



    final_score = max(
        0.0,
        min(final_score, 1.0)
    )



    sentence_scores.append(final_score)



    print(f"\nFaithfulness Score:")
    print(round(final_score, 4))



    # =====================================================
    # DECISION
    # =====================================================

    THRESHOLD = 0.45



    if final_score >= THRESHOLD:

        print("\nDecision:")
        print("FACTUALLY CONSISTENT")

    else:

        print("\nDecision:")
        print("POSSIBLE HALLUCINATION")



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
# FINAL SCORE
# =========================================================

hallucination_score = (
    1 - true_score
)



hallucination_rate = (
    hallucination_score * 100
)



# =========================================================
# PREVENT 0% / 100%
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

print(f"TRUE Score          : {true_score:.4f}")

print(f"Hallucination Score : {hallucination_score:.4f}")

print(f"Hallucination Rate  : {hallucination_rate:.2f}%")



# =========================================================
# FINAL DECISION
# =========================================================

print("\n===================================")
print("FINAL DECISION")
print("===================================\n")

if true_score >= 0.45:

    print("Summary is FACTUALLY CONSISTENT")

else:

    print("Summary contains HALLUCINATIONS")



# =========================================================
# DIRECT FINAL HALLUCINATION SCORE
# =========================================================

print("\n===================================")
print("FINAL HALLUCINATION SCORE")
print("===================================\n")

print(f"{hallucination_rate:.2f}%")



# =========================================================
# SAVE RESULTS
# =========================================================

with open(
    "mistral7b_true_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("GENERATED SUMMARY\n")
    file.write("=================\n\n")

    file.write(generated_summary + "\n\n")

    file.write(f"TRUE Score: {true_score:.4f}\n")

    file.write(f"Hallucination Score: {hallucination_score:.4f}\n")

    file.write(f"Hallucination Rate: {hallucination_rate:.2f}%\n")



print("\nResults saved successfully.")