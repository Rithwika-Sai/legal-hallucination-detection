## =========================================================
# FINAL STABLE QAFACTEVAL-STYLE LEGAL HALLUCINATION DETECTOR
# =========================================================
#
# FEATURES:
# ✅ BART Summarization
# ✅ Rule-based Legal Question Generation
# ✅ QA-based Faithfulness Checking
# ✅ Semantic Similarity Scoring
# ✅ Stable CPU Execution
# ✅ Hallucination Rate
# ✅ Clean Output
#
# =========================================================

# =========================================================
# HIDE WARNINGS
# =========================================================

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

# =========================================================
# IMPORTS
# =========================================================

import nltk
import numpy as np

from transformers import pipeline
from transformers.utils import logging

from sentence_transformers import (
    SentenceTransformer,
    util
)

# Hide transformers warnings
logging.set_verbosity_error()

# =========================================================
# DOWNLOAD TOKENIZER
# =========================================================

nltk.download('punkt', quiet=True)

# =========================================================
# LOAD MODELS
# =========================================================

# -----------------------------------------
# BART SUMMARIZER
# -----------------------------------------

summarizer = pipeline(
    "summarization",
    model="google/pegasus-xsum",
    tokenizer="google/pegasus-xsum",
    device=-1
)

# -----------------------------------------
# QA MODEL
# -----------------------------------------

qa_model = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=-1
)

# -----------------------------------------
# SEMANTIC EMBEDDING MODEL
# -----------------------------------------

embed_model = SentenceTransformer(
    'law-ai/InLegalBERT'
)

# =========================================================
# INPUT LEGAL DOCUMENT
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
# GENERATE SUMMARY USING BART
# =========================================================

summary_output = summarizer(
    source_document,
    max_length=60,
    min_length=20,
    do_sample=False,
    truncation=True
)

generated_summary = summary_output[0]["summary_text"]

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
# RULE-BASED LEGAL QUESTION GENERATION
# =========================================================

def generate_question(sentence):

    sentence_lower = sentence.lower()

    if "article" in sentence_lower:
        return "Which article is mentioned?"

    elif "section" in sentence_lower:
        return "Which section is mentioned?"

    elif "appeal" in sentence_lower:
        return "What happened to the appeal?"

    elif "petition" in sentence_lower:
        return "What petition was filed?"

    elif "court" in sentence_lower:
        return "What did the court decide?"

    elif "cost" in sentence_lower:
        return "What costs were imposed?"

    else:
        return "What is stated?"

# =========================================================
# QAFACTEVAL ANALYSIS
# =========================================================

faithfulness_scores = []

print("\n===================================")
print("QAFACTEVAL ANALYSIS")
print("===================================\n")

# =========================================================
# PROCESS EACH SUMMARY SENTENCE
# =========================================================

for idx, sent in enumerate(summary_sentences):

    # Skip tiny sentences
    if len(sent.strip()) < 10:
        continue

    print(f"\nSentence {idx+1}:")
    print(sent)

    # =====================================================
    # GENERATE QUESTION
    # =====================================================

    question = generate_question(sent)

    print(f"\nGenerated Question:")
    print(question)

    # =====================================================
    # ANSWER QUESTION USING SOURCE DOCUMENT
    # =====================================================

    qa_result = qa_model(
        question=question,
        context=source_document
    )

    source_answer = qa_result['answer']

    qa_confidence = qa_result['score']

    print(f"\nAnswer From Source:")
    print(source_answer)

    # =====================================================
    # SEMANTIC COMPARISON
    # =====================================================

    sent_embedding = embed_model.encode(
        sent,
        convert_to_tensor=True
    )

    answer_embedding = embed_model.encode(
        source_answer,
        convert_to_tensor=True
    )

    semantic_score = util.cos_sim(
        sent_embedding,
        answer_embedding
    ).item()

    # =====================================================
    # FINAL FAITHFULNESS SCORE
    # =====================================================

    final_score = (
        0.2 * qa_confidence
        +
        0.8 * semantic_score
    )

    # Clamp score
    final_score = max(0, min(final_score, 1))

    faithfulness_scores.append(final_score)

    print(f"\nQA Confidence     : {qa_confidence:.4f}")
    print(f"Semantic Score    : {semantic_score:.4f}")
    print(f"Faithfulness Score: {final_score:.4f}")

# =========================================================
# FINAL SCORES
# =========================================================

overall_faithfulness = np.mean(
    faithfulness_scores
)

hallucination_score = (
    1 - overall_faithfulness
)

hallucination_rate = (
    hallucination_score * 100
)

# =========================================================
# PRINT FINAL RESULTS
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

THRESHOLD = 0.35

print("\n===================================")
print("FINAL DECISION")
print("===================================\n")

if overall_faithfulness >= THRESHOLD:
    print("Summary is FACTUALLY CONSISTENT")
else:
    print("Summary contains HALLUCINATIONS")

# =========================================================
# SAVE RESULTS
# =========================================================

with open(
    "qafacteval_results.txt",
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