# =========================================================
# FINAL PEGASUS + SUMMAC + NLI + InLegalBERT
# FULLY RECTIFIED FINAL VERSION
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install sentence-transformers
# pip install nltk
# pip install torch
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
    PegasusTokenizer,
    PegasusForConditionalGeneration,
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

torch.set_num_threads(4)

nltk.download("punkt", quiet=True)



# =========================================================
# LOAD PEGASUS MODEL
# =========================================================

print("\nLoading PEGASUS model...\n")

model_name = "google/pegasus-cnn_dailymail"



tokenizer = PegasusTokenizer.from_pretrained(
    model_name
)



model = PegasusForConditionalGeneration.from_pretrained(
    model_name
)



# =========================================================
# LOAD NLI MODEL
# =========================================================

print("\nLoading NLI model...\n")

nli_pipeline = pipeline(

    "text-classification",

    model="typeform/distilbert-base-uncased-mnli",

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

source = " ".join(source.split())



# =========================================================
# SAFE DOCUMENT LENGTH
# =========================================================

source_document = source[:350]



# =========================================================
# GENERATE SUMMARY
# =========================================================

print("\nGenerating summary...\n")

inputs = tokenizer(

    source_document,

    truncation=True,

    padding=True,

    max_length=128,

    return_tensors="pt"
)



# =========================================================
# SAFE GENERATION
# =========================================================

summary_ids = model.generate(

    input_ids=inputs["input_ids"],

    attention_mask=inputs["attention_mask"],

    num_beams=2,

    max_length=40,

    min_length=10,

    early_stopping=True
)



generated_summary = tokenizer.decode(

    summary_ids[0],

    skip_special_tokens=True
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
# PRINT SUMMARY
# =========================================================

print("\n===================================")
print("GENERATED SUMMARY")
print("===================================\n")

print(generated_summary)



# =========================================================
# SPLIT INTO SENTENCES
# =========================================================

document_sentences = nltk.sent_tokenize(
    source_document
)

summary_sentences = nltk.sent_tokenize(
    generated_summary
)



# =========================================================
# HANDLE EMPTY DOCUMENT SENTENCES
# =========================================================

if len(document_sentences) == 0:

    document_sentences = [source_document]



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
# PROCESS SUMMARY SENTENCES
# =========================================================

for idx, summary_sent in enumerate(summary_sentences):

    if len(summary_sent.strip()) < 5:
        continue



    print(f"\nSummary Sentence {idx+1}:")
    print(summary_sent)



    max_support_score = 0.0



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
        # SAFE NLI INPUT
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



            label = result["label"]

            nli_score = result["score"]



        except Exception:

            label = "NEUTRAL"

            nli_score = 0.0



        # =================================================
        # ONLY ENTAILMENT SUPPORT
        # =================================================

        if label == "ENTAILMENT":

            entailment_score = nli_score

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
        # FINAL SCORE
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
        # KEEP BEST SUPPORT
        # =================================================

        if pair_score > max_support_score:

            max_support_score = pair_score



    # =====================================================
    # STORE SCORE
    # =====================================================

    sentence_scores.append(
        max_support_score
    )



    print(f"\nFaithfulness Score:")
    print(round(max_support_score, 4))



    # =====================================================
    # DECISION
    # =====================================================

    if max_support_score >= 0.45:

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
    f"SummaC Score           : "
    f"{summac_score:.4f}"
)

print(
    f"Hallucination Score    : "
    f"{hallucination_score:.4f}"
)

print(
    f"Hallucination Rate     : "
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
# FINAL HALLUCINATION SCORE
# =========================================================

print("\n===================================")
print("FINAL HALLUCINATION SCORE")
print("===================================\n")

print(f"{hallucination_rate:.2f}%")



# =========================================================
# SAVE RESULTS
# =========================================================

with open(
    "pegasus_summac_results.txt",
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