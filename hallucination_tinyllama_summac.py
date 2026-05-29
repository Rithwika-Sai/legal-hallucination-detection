# =========================================================
# FINAL SUMMAC + NLI + InLegalBERT + TinyLlama
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install sentence-transformers
# pip install nltk
# pip install torch
# pip install accelerate
# pip install sentencepiece

# =========================================================
# IMPORTS
# =========================================================

import os
import warnings
import torch
import nltk
import numpy as np

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

warnings.filterwarnings("ignore")

# =========================================================
# DOWNLOAD TOKENIZER
# =========================================================

nltk.download("punkt", quiet=True)

# =========================================================
# LOAD TinyLlama MODEL
# =========================================================

print("\nLoading TinyLlama model...\n")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
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
# SOURCE LEGAL DOCUMENT
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

source_document = source[:3000]

# =========================================================
# GENERATE SUMMARY USING TinyLlama
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
    max_length=1024
)

outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    temperature=0.3,
    do_sample=True,
    top_p=0.9
)

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

    print(f"\nSummary Sentence {idx+1}:")
    print(summary_sent)

    max_entailment = 0

    # =====================================================
    # COMPARE AGAINST EACH DOCUMENT SENTENCE
    # =====================================================

    for doc_sent in document_sentences:

        # =================================================
        # NLI INPUT
        # =================================================

        nli_input = f"""
        premise: {doc_sent}
        hypothesis: {summary_sent}
        """

        result = nli_pipeline(
            nli_input
        )[0]

        label = result['label']
        score = result['score']

        # =================================================
        # ONLY USE ENTAILMENT
        # =================================================

        if label == "ENTAILMENT":

            entailment_score = score

        else:

            entailment_score = 0

        # =================================================
        # SEMANTIC SIMILARITY
        # =================================================

        doc_embedding = embed_model.encode(
            doc_sent,
            convert_to_tensor=True
        )

        summary_embedding = embed_model.encode(
            summary_sent,
            convert_to_tensor=True
        )

        semantic_score = util.cos_sim(
            doc_embedding,
            summary_embedding
        ).item()

        # =================================================
        # FINAL PAIR SCORE
        # =================================================

        pair_score = (
            0.7 * entailment_score
            +
            0.3 * semantic_score
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

    THRESHOLD = 0.75

    if max_entailment >= THRESHOLD:

        print("\nDecision:")
        print("FACTUALLY CONSISTENT")

    else:

        print("\nDecision:")
        print("POSSIBLE HALLUCINATION")

# =========================================================
# FINAL SUMMAC SCORE
# =========================================================

summac_score = np.mean(
    sentence_scores
)

hallucination_score = (
    1 - summac_score
)

hallucination_rate = (
    hallucination_score * 100
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

if summac_score >= 0.75:

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
    "summac_tinyllama_results.txt",
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