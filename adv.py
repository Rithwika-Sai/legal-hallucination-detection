# =========================================================
# FINAL HYBRID QA + NLI HALLUCINATION DETECTOR
# =========================================================
# Features:
# ✅ Multi-QA + NLI combined
# ✅ Retrieval-based verification
# ✅ Soft hallucination scoring
# ✅ Legal-aware question generation
# ✅ CPU-friendly
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
# =========================================================
# INLEGALBERT EMBEDDING MODEL
# =========================================================

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

with open("outputs/bart_output.txt", "r", encoding="utf-8") as f:
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
# ADVANCED LEGAL QUESTION GENERATOR
# =========================================================

def generate_questions(sentence):

    questions = []

    sentence = sentence.lower()

    # =====================================================
    # 1. ACTOR QUESTIONS
    # =====================================================

    actor_keywords = [
        "court",
        "judge",
        "appellant",
        "respondent",
        "petitioner",
        "plaintiff",
        "defendant"
    ]

    if any(word in sentence for word in actor_keywords):

        questions.extend([

            "Who performed the action?",

            "Which party is involved?",

            "Which court made the decision?"
        ])

    # =====================================================
    # 2. ACTION QUESTIONS
    # =====================================================

    action_keywords = [
        "dismissed",
        "allowed",
        "granted",
        "rejected",
        "affirmed",
        "reversed",
        "ordered",
        "held"
    ]

    if any(word in sentence for word in action_keywords):

        questions.extend([

            "What action was taken?",

            "What did the court decide?",

            "What decision was made?"
        ])

    # =====================================================
    # 3. OBJECT QUESTIONS
    # =====================================================

    object_keywords = [
        "appeal",
        "petition",
        "application",
        "case",
        "suit",
        "writ"
    ]

    if any(word in sentence for word in object_keywords):

        questions.extend([

            "What petition or appeal is discussed?",

            "What object was affected?",

            "Which legal matter is involved?"
        ])

    # =====================================================
    # 4. LEGAL BASIS QUESTIONS
    # =====================================================

    legal_keywords = [
        "section",
        "article",
        "constitution",
        "cpc",
        "ipc",
        "act"
    ]

    if any(word in sentence for word in legal_keywords):

        questions.extend([

            "Which section is mentioned?",

            "Under which article or section?",

            "What legal provision was cited?",

            "What statute is referenced?"
        ])

    # =====================================================
    # 5. OUTCOME QUESTIONS
    # =====================================================

    outcome_keywords = [
        "dismissed",
        "allowed",
        "disposed",
        "granted",
        "denied",
        "convicted",
        "acquitted",
        "remanded"
    ]

    if any(word in sentence for word in outcome_keywords):

        questions.extend([

            "What was the final outcome?",

            "Was the appeal allowed or dismissed?",

            "What happened finally?"
        ])

    # =====================================================
    # FALLBACK QUESTION
    # =====================================================

    if len(questions) == 0:

        questions.append(
            "What is stated in the sentence?"
        )

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    questions = list(set(questions))

    return questions

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

    confidence = 0

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

        # -------------------------------------------------
        # ENTAILMENT
        # -------------------------------------------------

        if (
            label == "ENTAILMENT"
            and confidence > 0.25
        ):

            entailment_found = True

        # -------------------------------------------------
        # CONTRADICTION
        # -------------------------------------------------

        if (
            label == "CONTRADICTION"
            and confidence > 0.80
        ):

            contradiction_found = True

    # =====================================================
    # MULTI-QA CHECK
    # =====================================================

    questions = generate_questions(sent)

    context = " ".join(evidence_sentences)

    context = context[:400]

    qa_scores = []

    best_answer = ""

    for question in questions:

        try:

            qa_result = qa_model(
                question=question,
                context=context
            )

            answer = qa_result['answer']

            qa_conf = qa_result['score']

            # -------------------------------------------------
            # SEMANTIC MATCH
            # -------------------------------------------------

            answer_embedding = embed_model.encode(
                answer,
                convert_to_tensor=True
            )

            answer_score = util.cos_sim(
                answer_embedding,
                sent_embedding
            ).item()

            # -------------------------------------------------
            # COMBINED QA SCORE
            # -------------------------------------------------

            combined_score = (
                (qa_conf * 0.5)
                +
                (answer_score * 0.5)
            )

            qa_scores.append(combined_score)

            if combined_score == max(qa_scores):

                best_answer = answer

        except:

            continue

    # =====================================================
    # FINAL QA SCORE
    # =====================================================

    if len(qa_scores) > 0:

        final_qa_score = max(qa_scores)

    else:

        final_qa_score = 0.0

    # =====================================================
    # FINAL HALLUCINATION SCORE
    # =====================================================

    sentence_score = 0

    # Strong contradiction
    if contradiction_found:

        sentence_score += confidence * 0.6

    # Weak/no entailment
    if not entailment_found:

        sentence_score += (1 - confidence) * 0.3

    # QA uncertainty
    sentence_score += (1 - final_qa_score) * 0.4

    # Very weak QA match
    if final_qa_score < 0.20:

        sentence_score += 0.15

    # Limit score
    sentence_score = min(sentence_score, 1.0)

    # Add to total
    hallucinated += sentence_score

    # =====================================================
    # LABEL
    # =====================================================

    if sentence_score >= 0.75:

        hall_label = "HIGH HALLUCINATION"

    elif sentence_score >= 0.45:

        hall_label = "MODERATE HALLUCINATION"

    else:

        hall_label = "LOW / SUPPORTED"

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print("\n----------------------------------")
    print(f"SENTENCE: {sent}")
    print("----------------------------------")

    print("\nQUESTIONS:")
    for q in questions:

        print(f"- {q}")

    print("\nBEST ANSWER:")
    print(best_answer)

    print(f"\nFINAL QA SCORE: {final_qa_score:.4f}")

    print(f"HALLUCINATION SCORE: {sentence_score:.4f}")

    print(f"LABEL: {hall_label}")

    print("----------------------------------")

# =========================================================
# FINAL HALLUCINATION RATE
# =========================================================

total_sentences = len(summary_sentences)

hallucination_rate = (
    hallucinated / total_sentences
) * 100

# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n======================================")
print("FINAL SUMMARY")
print("======================================\n")

print(summary)

print("\n======================================")
print(
    f"FINAL HALLUCINATION RATE: "
    f"{hallucination_rate:.2f}%"
)
print("======================================")