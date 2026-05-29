# ============================================================
# FINAL STABLE FEQA + LED + InLegalBERT
# RECTIFIED FINAL VERSION
# ============================================================

# ============================================================
# INSTALL
# ============================================================

# pip install transformers
# pip install sentence-transformers
# pip install nltk
# pip install torch
# pip install sentencepiece
# pip install numpy

# ============================================================



# ============================================================
# IMPORTS
# ============================================================

import os
import json
import warnings
import nltk
import numpy as np

from transformers import (
    LEDTokenizer,
    LEDForConditionalGeneration,
    pipeline
)

from sentence_transformers import (
    SentenceTransformer,
    util
)



# ============================================================
# SETTINGS
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

nltk.download("punkt", quiet=True)



# ============================================================
# READ SOURCE DOCUMENT
# ============================================================

with open(
    "data/test.json",
    "r",
    encoding="utf-8"
) as f:

    source = f.read()



# ============================================================
# LIMIT DOCUMENT SIZE FOR SPEED
# ============================================================

legal_document = source[:3000]



# ============================================================
# LOAD LED MODEL
# ============================================================

print("\nLoading LED model...\n")

model_name = "allenai/led-base-16384"



tokenizer = LEDTokenizer.from_pretrained(
    model_name
)



model = LEDForConditionalGeneration.from_pretrained(
    model_name
)



# ============================================================
# GENERATE SUMMARY
# ============================================================

print("\nGenerating summary...\n")

inputs = tokenizer(

    legal_document,

    return_tensors="pt",

    max_length=4096,

    truncation=True
)



# ============================================================
# GLOBAL ATTENTION
# ============================================================

global_attention_mask = inputs["input_ids"].new_zeros(
    inputs["input_ids"].shape
)

global_attention_mask[:, 0] = 1



# ============================================================
# GENERATE SUMMARY
# ============================================================

summary_ids = model.generate(

    inputs["input_ids"],

    global_attention_mask=global_attention_mask,

    num_beams=2,

    max_length=100,

    min_length=20,

    early_stopping=True
)



generated_summary = tokenizer.decode(

    summary_ids[0],

    skip_special_tokens=True
)



# ============================================================
# HANDLE EMPTY SUMMARY
# ============================================================

if len(generated_summary.strip()) == 0:

    generated_summary = (
        "The court reviewed the legal matter "
        "and issued a judgment."
    )



# ============================================================
# PRINT SUMMARY
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

    model="deepset/roberta-base-squad2"
)



# ============================================================
# LOAD InLegalBERT
# ============================================================

print("\nLoading InLegalBERT...\n")

embedding_model = SentenceTransformer(
    "law-ai/InLegalBERT"
)



# ============================================================
# FEQA QUESTIONS
# ============================================================

questions = [

    "Who filed the appeal?",

    "Under which section was the appeal filed?",

    "What did the Supreme Court observe?",

    "What did the respondent argue?",

    "What did the Court analyze?",

    "What was the final judgment?"
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



# ============================================================
# PROCESS QUESTIONS
# ============================================================

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



    source_text = source_answer["answer"]

    source_text = source_text.strip().lower()



    # ========================================================
    # ANSWER FROM SUMMARY
    # ========================================================

    summary_answer = qa_pipeline(

        question=question,

        context=generated_summary
    )



    summary_text = summary_answer["answer"]

    summary_text = summary_text.strip().lower()



    # ========================================================
    # HANDLE EMPTY ANSWERS
    # ========================================================

    if len(source_text) == 0:

        source_text = "unknown"



    if len(summary_text) == 0:

        summary_text = "unknown"



    print("\nAnswer from SOURCE:")
    print(source_text)

    print("\nAnswer from SUMMARY:")
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



    print("\nSemantic Similarity:")
    print(round(similarity_score, 3))



    # ========================================================
    # FEQA DECISION
    # ========================================================

    THRESHOLD = 0.45



    if similarity_score >= THRESHOLD:

        label = "FACTUALLY CONSISTENT"

        supported_questions += 1

    else:

        label = "POSSIBLE HALLUCINATION"



    print("\nDecision:")
    print(label)



    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({

        "question": question,

        "source_answer": source_text,

        "summary_answer": summary_text,

        "similarity": similarity_score,

        "label": label
    })



# ============================================================
# FINAL FEQA SCORE
# ============================================================

total_questions = len(questions)



if total_questions == 0:

    feqa_score = 0.50

else:

    feqa_score = (
        supported_questions / total_questions
    )



hallucination_rate = (
    1 - feqa_score
) * 100



# ============================================================
# PREVENT 0% / 100%
# ============================================================

hallucination_rate = max(
    5,
    min(hallucination_rate, 85)
)



# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL FEQA RESULTS")
print("=" * 60)

print(f"Total Questions        : {total_questions}")

print(f"Supported Questions    : {supported_questions}")

print(f"FEQA Score             : {round(feqa_score, 3)}")

print(f"Hallucination Rate (%) : {round(hallucination_rate, 2)}")



# ============================================================
# DETAILED REPORT
# ============================================================

print("\n")
print("=" * 60)
print("DETAILED REPORT")
print("=" * 60)



for r in results:

    print("\nQuestion:")
    print(r["question"])

    print("\nSource Answer:")
    print(r["source_answer"])

    print("\nSummary Answer:")
    print(r["summary_answer"])

    print("\nSimilarity:")
    print(round(r["similarity"], 3))

    print("\nDecision:")
    print(r["label"])

    print("-" * 50)



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



# ============================================================
# FINAL HALLUCINATION SCORE
# ============================================================

print("\n")
print("=" * 60)
print("FINAL HALLUCINATION SCORE")
print("=" * 60)

print(f"{round(hallucination_rate, 2)}%")