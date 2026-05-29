# ============================================================
# FAST LEGAL HALLUCINATION DETECTION PIPELINE
# FLAN-T5 + FEQA + Semantic Similarity
# VERY FAST VERSION
# ============================================================

# ============================================================
# INSTALL
# ============================================================

# pip install transformers
# pip install sentence-transformers
# pip install nltk
# pip install torch

# ============================================================
# IMPORTS
# ============================================================

import json
import nltk
import torch

from transformers import pipeline

from sentence_transformers import (

    SentenceTransformer,

    util
)

# ============================================================
# DOWNLOAD NLTK
# ============================================================

nltk.download("punkt", quiet=True)

# ============================================================
# LOAD JSON FILE
# ============================================================

with open(
    "data/test.json",
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)

# ============================================================
# TAKE ONLY 1 SAMPLE
# ============================================================

sample = data[0]

legal_document = sample["judgment"]

reference_summary = sample["reference_summary"]

# ============================================================
# LIMIT INPUT SIZE
# ============================================================

legal_document = legal_document[:300]

# ============================================================
# LOAD FAST FLAN-T5 MODEL
# ============================================================

print("\nLoading FLAN-T5...\n")

summarizer = pipeline(

    "text2text-generation",

    model="google/flan-t5-base",

    device=-1
)

# ============================================================
# GENERATE SUMMARY
# ============================================================

print("\nGenerating summary...\n")

prompt = f"""
Summarize this legal judgment briefly:

{legal_document}
"""

output = summarizer(

    prompt,

    max_length=80,

    do_sample=False
)

generated_summary = output[0]["generated_text"]

# ============================================================
# CLEAN SUMMARY
# ============================================================

generated_summary = generated_summary.replace(
    "\n",
    " "
)

generated_summary = " ".join(
    generated_summary.split()
)

# ============================================================
# HANDLE EMPTY SUMMARY
# ============================================================

if len(generated_summary.strip()) == 0:

    generated_summary = (
        "The court reviewed the legal matter "
        "and delivered the judgment."
    )

# ============================================================
# PRINT GENERATED SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("GENERATED SUMMARY")
print("=" * 60)

print(generated_summary)

# ============================================================
# LOAD QA MODEL
# ============================================================

print("\nLoading QA model...\n")

qa_pipeline = pipeline(

    "question-answering",

    model="distilbert-base-cased-distilled-squad"
)

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...\n")

embedding_model = SentenceTransformer(

    "all-MiniLM-L6-v2"
)

# ============================================================
# FEQA QUESTIONS
# ============================================================

questions = [

    "Who filed the appeal?",

    "What was the final judgment?",

    "What legal issue was discussed?"
]

# ============================================================
# FEQA VERIFICATION
# ============================================================

supported_questions = 0

results = []

print("\n")
print("=" * 60)
print("FEQA FACTUAL VERIFICATION")
print("=" * 60)

for idx, question in enumerate(questions):

    print("\n")
    print("=" * 60)
    print(f"QUESTION {idx+1}")
    print("=" * 60)

    print("\nQuestion:")
    print(question)

    # ========================================================
    # ANSWER FROM SOURCE
    # ========================================================

    source_answer = qa_pipeline(

        question=question,

        context=legal_document
    )

    source_text = source_answer["answer"].lower()

    # ========================================================
    # ANSWER FROM SUMMARY
    # ========================================================

    summary_answer = qa_pipeline(

        question=question,

        context=generated_summary
    )

    summary_text = summary_answer["answer"].lower()

    print("\nSource Answer:")
    print(source_text)

    print("\nSummary Answer:")
    print(summary_text)

    # ========================================================
    # SEMANTIC SIMILARITY
    # ========================================================

    embeddings = embedding_model.encode(

        [source_text, summary_text],

        convert_to_tensor=True
    )

    similarity_score = util.cos_sim(

        embeddings[0],

        embeddings[1]

    ).item()

    print("\nSimilarity Score:")
    print(round(similarity_score, 3))

    # ========================================================
    # DECISION
    # ========================================================

    if similarity_score >= 0.55:

        label = "FACTUALLY CONSISTENT"

        supported_questions += 1

    else:

        label = "POSSIBLE HALLUCINATION"

    print("\nDecision:")
    print(label)

# ============================================================
# FINAL SCORES
# ============================================================

total_questions = len(questions)

feqa_score = supported_questions / total_questions

hallucination_rate = (
    1 - feqa_score
) * 100

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"FEQA Score             : {round(feqa_score, 3)}")

print(f"Hallucination Rate (%) : {round(hallucination_rate, 2)}")

# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n")
print("=" * 60)
print("FINAL INTERPRETATION")
print("=" * 60)

if feqa_score >= 0.90:

    print("Excellent factual consistency.")

elif feqa_score >= 0.75:

    print("Good factual consistency.")

elif feqa_score >= 0.50:

    print("Moderate hallucination detected.")

else:

    print("High hallucination detected.")


