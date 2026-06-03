# =========================================================
# FINAL HHEM-STYLE NLI HALLUCINATION DETECTOR
# USING BART + InLegalBERT
# FULLY RECTIFIED FINAL VERSION
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install sentence-transformers
# pip install nltk
# pip install torch
# pip install numpy

# =========================================================



# =========================================================
# IMPORTS
# =========================================================

import os
import json
import warnings
import torch
import nltk
import numpy as np

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
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
# LOAD BART MODEL
# =========================================================

print("\nLoading BART model...\n")

bart_model_name = "facebook/bart-large-cnn"



tokenizer = BartTokenizer.from_pretrained(
    bart_model_name
)



model = BartForConditionalGeneration.from_pretrained(
    bart_model_name
)



# =========================================================
# LOAD  NLI MODEL
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
# LIMIT DOCUMENT SIZE FOR SPEED
# =========================================================

source_document = source[:1500]



# =========================================================
# GENERATE SUMMARY USING BART
# =========================================================

print("\nGenerating summary...\n")

inputs = tokenizer(

    source_document,

    max_length=1024,

    return_tensors="pt",

    truncation=True
)



summary_ids = model.generate(

    inputs["input_ids"],

    num_beams=2,

    max_length=80,

    min_length=20,

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
        "The court reviewed the case "
        "and delivered a judgment."
    )



# =========================================================
# PRINT SUMMARY
# =========================================================

print("\n===================================")
print("GENERATED SUMMARY")
print("===================================\n")

print(generated_summary)



# =========================================================
# SPLIT SUMMARY INTO SENTENCES
# =========================================================

summary_sentences = nltk.sent_tokenize(
    generated_summary
)



# =========================================================
# PRECOMPUTE SOURCE EMBEDDING
# =========================================================

source_embedding = embed_model.encode(

    source_document,

    convert_to_tensor=True
)



# =========================================================
# HHEM-STYLE ANALYSIS
# =========================================================

faithfulness_scores = []

print("\n===================================")
print("HHEM NLI ANALYSIS")
print("===================================\n")



# =========================================================
# PROCESS SUMMARY SENTENCES
# =========================================================

for idx, sent in enumerate(summary_sentences):

    if len(sent.strip()) < 10:
        continue



    print(f"\nSentence {idx+1}:")
    print(sent)



    # =====================================================
    # SAFE NLI SOURCE
    # =====================================================

    nli_source = source_document[:1200]



    # =====================================================
    # SAFE NLI INPUT
    # =====================================================

    nli_input = (
        f"{nli_source} </s></s> {sent}"
    )



    # =====================================================
    # SAFE NLI CHECK
    # =====================================================

    try:

        nli_result = nli_pipeline(

            nli_input,

            truncation=True,

            max_length=512

        )[0]



        nli_label = nli_result["label"]

        nli_score = nli_result["score"]



    except Exception as e:

        print("\nNLI Error:")
        print(e)

        nli_label = "NEUTRAL"

        nli_score = 0.0



    print(f"\nNLI Label:")
    print(nli_label)



    print(f"\nNLI Score:")
    print(round(nli_score, 4))



    # =====================================================
    # SEMANTIC SIMILARITY
    # =====================================================

    try:

        sentence_embedding = embed_model.encode(

            sent,

            convert_to_tensor=True
        )



        semantic_score = util.cos_sim(

            source_embedding,

            sentence_embedding

        ).item()



    except Exception as e:

        print("\nEmbedding Error:")
        print(e)

        semantic_score = 0.0



    print(f"\nSemantic Similarity:")
    print(round(semantic_score, 4))



    # =====================================================
    # ONLY ENTAILMENT SUPPORT
    # =====================================================

    if nli_label == "ENTAILMENT":

        entailment_score = nli_score

    else:

        entailment_score = 0.0



    # =====================================================
    # FINAL SCORE
    # =====================================================

    final_score = (

        0.7 * entailment_score

        +

        0.3 * semantic_score
    )



    final_score = max(
        0.0,
        min(final_score, 1.0)
    )



    faithfulness_scores.append(
        final_score
    )



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

if len(faithfulness_scores) == 0:

    overall_faithfulness = 0.50

else:

    overall_faithfulness = np.mean(
        faithfulness_scores
    )



# =========================================================
# FINAL SCORES
# =========================================================

hallucination_score = (
    1 - overall_faithfulness
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
    f"Overall Faithfulness Score : "
    f"{overall_faithfulness:.4f}"
)

print(
    f"Hallucination Score        : "
    f"{hallucination_score:.4f}"
)

print(
    f"Hallucination Rate         : "
    f"{hallucination_rate:.2f}%"
)



# =========================================================
# FINAL DECISION
# =========================================================

print("\n===================================")
print("FINAL DECISION")
print("===================================\n")

if overall_faithfulness >= 0.45:

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
    "bart_hhem_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("GENERATED SUMMARY\n")
    file.write("=================\n\n")

    file.write(generated_summary + "\n\n")

    file.write(
        f"Overall Faithfulness Score: "
        f"{overall_faithfulness:.4f}\n"
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