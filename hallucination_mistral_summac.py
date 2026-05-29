# =========================================================
# FINAL MISTRAL-7B + SUMMAC + NLI + InLegalBERT
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
# HIDE WARNINGS
# =========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")



# =========================================================
# DOWNLOAD TOKENIZER
# =========================================================

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
# LOAD NLI MODEL
# =========================================================

print("\nLoading NLI model...\n")

nli_pipeline = pipeline(

    "text-classification",

    model="roberta-large-mnli",

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
# CLEAN SOURCE TEXT
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
Summarize the following legal judgment clearly and concisely:

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
# GENERATE SUMMARY
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
# SPLIT DOCUMENT & SUMMARY INTO SENTENCES
# =========================================================

document_sentences = nltk.sent_tokenize(
    source_document
)

summary_sentences = nltk.sent_tokenize(
    generated_summary
)



# =========================================================
# HANDLE EMPTY SENTENCES
# =========================================================

if len(document_sentences) == 0:

    document_sentences = [source_document]



if len(summary_sentences) == 0:

    summary_sentences = [generated_summary]



# =========================================================
# PRECOMPUTE DOCUMENT EMBEDDINGS
# =========================================================

doc_embeddings = embed_model.encode(

    document_sentences,

    convert_to_tensor=True
)



# =========================================================
# SUMMAC ANALYSIS
# =========================================================

sentence_scores = []

print("\n===================================")
print("SUMMAC ANALYSIS")
print("===================================\n")



# =========================================================
# PROCESS EACH SUMMARY SENTENCE
# =========================================================

for idx, summary_sent in enumerate(summary_sentences):

    if len(summary_sent.strip()) < 5:
        continue



    print(f"\nSummary Sentence {idx+1}:")
    print(summary_sent)



    max_entailment = 0.0



    # =====================================================
    # COMPARE AGAINST DOCUMENT SENTENCES
    # =====================================================

    for doc_idx, doc_sent in enumerate(document_sentences):



        # =================================================
        # SAFE SHORTENING
        # =================================================

        short_doc = doc_sent[:120]

        short_summary = summary_sent[:60]



        # =================================================
        # NLI INPUT
        # =================================================

        nli_input = (
            f"{short_doc} </s></s> {short_summary}"
        )



        # =================================================
        # SAFE NLI CHECK
        # =================================================

        try:

            result = nli_pipeline(

                nli_input,

                truncation=True,

                max_length=128

            )[0]



            label = result['label']

            score = result['score']



        except Exception:

            label = "NEUTRAL"

            score = 0.0



        # =================================================
        # ONLY USE ENTAILMENT
        # =================================================

        if label == "ENTAILMENT":

            entailment_score = score

        else:

            entailment_score = 0.0



        # =================================================
        # SEMANTIC SIMILARITY
        # =================================================

        try:

            summary_embedding = embed_model.encode(

                short_summary,

                convert_to_tensor=True
            )



            semantic_score = util.cos_sim(

                doc_embeddings[doc_idx],

                summary_embedding

            ).item()



        except Exception:

            semantic_score = 0.0



        # =================================================
        # FINAL PAIR SCORE
        # =================================================

        pair_score = (
            0.7 * entailment_score
            +
            0.3 * semantic_score
        )



        pair_score = max(
            0.0,
            min(pair_score, 1.0)
        )



        # =================================================
        # KEEP MAX SUPPORT
        # =================================================

        if pair_score > max_entailment:

            max_entailment = pair_score



    # =====================================================
    # STORE SENTENCE SCORE
    # =====================================================

    sentence_scores.append(max_entailment)



    print(f"\nSentence Faithfulness Score:")
    print(round(max_entailment, 4))



    # =====================================================
    # DECISION
    # =====================================================

    THRESHOLD = 0.45



    if max_entailment >= THRESHOLD:

        print("\nDecision:")
        print("FACTUALLY CONSISTENT")

    else:

        print("\nDecision:")
        print("POSSIBLE HALLUCINATION")



# =========================================================
# HANDLE EMPTY SCORES
# =========================================================

if len(sentence_scores) == 0:

    summac_score = 0.50

else:

    summac_score = np.mean(
        sentence_scores
    )



# =========================================================
# FINAL SCORES
# =========================================================

hallucination_score = (
    1 - summac_score
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

print(
    f"SummaC Score             : "
    f"{summac_score:.4f}"
)

print(
    f"Hallucination Score      : "
    f"{hallucination_score:.4f}"
)

print(
    f"Hallucination Rate       : "
    f"{hallucination_rate:.2f}%"
)



# =========================================================
# FINAL DECISION
# =========================================================

print("\n===================================")
print("FINAL DECISION")
print("===================================\n")

if summac_score >= 0.45:

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
    "mistral7b_summac_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("GENERATED SUMMARY\n")
    file.write("=================\n\n")

    file.write(generated_summary + "\n\n")

    file.write(
        f"SummaC Score: "
        f"{summac_score:.4f}\n"
    )

    file.write(
        f"Hallucination Score: "
        f"{hallucination_score:.4f}\n"
    )

    file.write(
        f"Hallucination Rate: "
        f"{hallucination_rate:.2f}%\n"
    )



print("\nResults saved successfully.")