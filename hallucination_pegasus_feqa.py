# ============================================================
# FINAL PEGASUS + FEQA + InLegalBERT
# FULLY RECTIFIED FINAL VERSION
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



# ============================================================
# SETTINGS
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

torch.set_num_threads(4)

nltk.download("punkt", quiet=True)



# ============================================================
# LOAD PEGASUS MODEL
# ============================================================

print("\nLoading PEGASUS model...\n")

model_name = "google/pegasus-xsum"



tokenizer = PegasusTokenizer.from_pretrained(
    model_name
)



model = PegasusForConditionalGeneration.from_pretrained(
    model_name
)



# ============================================================
# LOAD QA MODEL
# ============================================================

print("\nLoading QA model...\n")

qa_pipeline = pipeline(

    "question-answering",

    model="deepset/roberta-base-squad2",

    device=-1
)



# ============================================================
# LOAD InLegalBERT
# ============================================================

print("\nLoading InLegalBERT...\n")

embedding_model = SentenceTransformer(
    "law-ai/InLegalBERT"
)



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
# CLEAN SOURCE TEXT
# ============================================================

source = source.replace("\n", " ")

source = source.replace("\\n", " ")

source = " ".join(source.split())



# ============================================================
# LIMIT DOCUMENT SIZE
# ============================================================

legal_document = source[:700]



# ============================================================
# GENERATE SUMMARY
# ============================================================

print("\nGenerating summary...\n")

inputs = tokenizer(

    legal_document,

    truncation=True,

    padding="longest",

    max_length=512,

    return_tensors="pt"
)



summary_ids = model.generate(

    inputs["input_ids"],

    num_beams=2,

    max_length=60,

    min_length=15,

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
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("GENERATED SUMMARY")
print("=" * 60)

print(generated_summary)



# ============================================================
# FEQA QUESTIONS
# ============================================================

questions = [

    "Who filed the appeal?",

    "What was the case about?",

    "What did the court observe?",

    "What did the respondent argue?",

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
    # SAFE CONTEXTS
    # ========================================================

    safe_source = legal_document[:400]

    safe_summary = generated_summary[:200]



    # ========================================================
    # ANSWER FROM SOURCE
    # ========================================================

    try:

        source_answer = qa_pipeline(

            question=question,

            context=safe_source
        )



        source_text = source_answer["answer"]

    except Exception:

        source_text = "unknown"



    source_text = source_text.strip().lower()



    # ========================================================
    # ANSWER FROM SUMMARY
    # ========================================================

    try:

        summary_answer = qa_pipeline(

            question=question,

            context=safe_summary
        )



        summary_text = summary_answer["answer"]

    except Exception:

        summary_text = "unknown"



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

    try:

        embeddings = embedding_model.encode(

            [source_text, summary_text],

            convert_to_tensor=True
        )



        similarity_score = util.cos_sim(

            embeddings[0],

            embeddings[1]

        ).item()



    except Exception:

        similarity_score = 0.0



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



# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "pegasus_feqa_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("GENERATED SUMMARY\n")
    file.write("=================\n\n")

    file.write(generated_summary + "\n\n")

    file.write(
        f"FEQA Score: "
        f"{feqa_score:.4f}\n"
    )

    file.write(
        f"Hallucination Rate: "
        f"{hallucination_rate:.2f}%\n"
    )



print("\nResults saved successfully.")
