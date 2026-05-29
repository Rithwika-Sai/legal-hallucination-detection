# =========================================================
# FINAL HYBRID QA + NLI HALLUCINATION DETECTOR
# =========================================================
# Features:
# ✅ QA + NLI combined
# ✅ Avoids 0% and 100%
# ✅ Soft hallucination scoring
# ✅ Handles long legal docs
# ✅ CPU-friendly
# ✅ Prints ONLY summary + hall rate
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

# Hide transformer warnings
logging.set_verbosity_error()

# =========================================================
# DOWNLOAD TOKENIZER
# =========================================================

nltk.download('punkt', quiet=True)

# =========================================================
# LOAD MODELS
# =========================================================

# Embedding model
embed_model = SentenceTransformer(
    'law-ai/InLegalBERT'
)

# NLI model
nli_model = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    device=-1
)

# QA model
qa_model = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=-1
)

# =========================================================
# READ SOURCE DOCUMENT
# =========================================================

with open("data/test.json", "r", encoding="utf-8") as f:
    source = f.read()

# =========================================================
# READ GENERATED SUMMARY
# =========================================================

with open("outputs/led_output.txt", "r", encoding="utf-8") as f:
    summary = f.read()

# =========================================================
# LIMIT HUGE LEGAL DOCUMENT
# =========================================================

source = source[:50000]

# =========================================================
# SPLIT INTO SENTENCES
# =========================================================

source_sentences = nltk.sent_tokenize(source)

summary_sentences = nltk.sent_tokenize(summary)

# =========================================================
# ENCODE SOURCE SENTENCES
# =========================================================

source_embeddings = embed_model.encode(
    source_sentences,
    convert_to_tensor=True
)

# =========================================================
# SIMPLE QUESTION GENERATOR
# =========================================================

def generate_question(sentence):

    sentence = sentence.lower()

    if "court" in sentence:
        return "What did the court decide?"

    elif "appeal" in sentence:
        return "What happened to the appeal?"

    elif "section" in sentence:
        return "Which section is mentioned?"

    elif "article" in sentence:
        return "Which article is mentioned?"

    elif "petition" in sentence:
        return "What petition is discussed?"

    else:
        return "What is stated?"

# =========================================================
# HALLUCINATION SCORE
# =========================================================

hallucinated = 0

# =========================================================
# MAIN LOOP
# =========================================================

for sent in summary_sentences:

    # Skip very tiny sentences
    if len(sent.strip()) < 15:
        continue

    # -----------------------------------------------------
    # EMBEDDING
    # -----------------------------------------------------

    sent_embedding = embed_model.encode(
        sent,
        convert_to_tensor=True
    )

    # -----------------------------------------------------
    # RETRIEVAL
    # -----------------------------------------------------

    similarities = util.cos_sim(
        sent_embedding,
        source_embeddings
    )

    similarities = similarities.cpu().numpy()[0]

    top_k = 3

    top_indices = similarities.argsort()[-top_k:][::-1]

    evidence_sentences = [
        source_sentences[i]
        for i in top_indices
    ]

    # -----------------------------------------------------
    # NLI CHECK
    # -----------------------------------------------------

    entailment_found = False
    contradiction_found = False

    for evidence in evidence_sentences:

        evidence = evidence[:300]

        short_sent = sent[:180]

        result = nli_model(
            f"{evidence} </s></s> {short_sent}",
            truncation=True,
            max_length=512
        )[0]

        label = result['label']

        confidence = result['score']

        # ---------------------------------------------
        # ENTAILMENT
        # ---------------------------------------------

        if (
            label == "ENTAILMENT"
            and confidence > 0.25
        ):

            entailment_found = True

        # ---------------------------------------------
        # CONTRADICTION
        # ---------------------------------------------

        if (
            label == "CONTRADICTION"
            and confidence > 0.80
        ):

            contradiction_found = True

    # -----------------------------------------------------
    # QA CHECK
    # -----------------------------------------------------

    question = generate_question(sent)

    context = " ".join(evidence_sentences)

    context = context[:400]

    qa_result = qa_model(
        question=question,
        context=context
    )

    answer = qa_result['answer']

    qa_conf = qa_result['score']

    # -----------------------------------------------------
    # QA SEMANTIC MATCH
    # -----------------------------------------------------

    answer_embedding = embed_model.encode(
        answer,
        convert_to_tensor=True
    )

    answer_score = util.cos_sim(
        answer_embedding,
        sent_embedding
    ).item()

    # =====================================================
    # SOFT HALLUCINATION SCORE
    # =====================================================

    sentence_score = 0

    # Strong contradiction
    # =====================================================
# DYNAMIC HALLUCINATION SCORING
# =====================================================

# Strong contradiction
    if contradiction_found:

        sentence_score += confidence

# Weak/no entailment
    if not entailment_found:

        sentence_score += (1 - confidence) * 0.4

# QA uncertainty
    sentence_score += (1 - qa_conf) * 0.2

# Semantic mismatch
    sentence_score += (1 - answer_score) * 0.2

# VERY weak semantic match only
    if answer_score < 0.10:

        sentence_score += 0.10

    # Limit score
    sentence_score = min(sentence_score, 1.0)

    # Add to total
    hallucinated += sentence_score

# =========================================================
# FINAL HALLUCINATION RATE
# =========================================================

total_sentences = len(summary_sentences)

hallucination_rate = (
    hallucinated / total_sentences
) * 100

# =========================================================
# PRINT OUTPUT
# =========================================================

print("\n==============================")
print("SUMMARY")
print("==============================\n")

print(summary)

print("\n==============================")
print(
    f"HALLUCINATION RATE: "
    f"{hallucination_rate:.2f}%"
)
print("==============================")