# ============================================================
# FINAL STABLE FEQA + TinyLlama + InLegalBERT
# ============================================================

# ============================================================
# INSTALL
# ============================================================

# pip install transformers
# pip install sentence-transformers
# pip install nltk
# pip install torch
# pip install accelerate
# pip install sentencepiece

# ============================================================



# ============================================================
# IMPORTS
# ============================================================

import torch
import nltk

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from sentence_transformers import (
    SentenceTransformer,
    util
)

nltk.download("punkt")



# ============================================================
# INPUT LEGAL DOCUMENT
# ============================================================

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



# ============================================================
# LOAD TinyLlama MODEL
# ============================================================

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



# ============================================================
# GENERATE SUMMARY
# ============================================================

print("\nGenerating summary...\n")

prompt = f"""
Summarize the following legal judgment clearly and concisely:

{legal_document}

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



# ============================================================
# CLEAN SUMMARY
# ============================================================

generated_summary = generated_summary.replace(
    prompt,
    ""
).strip()



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
# MANUAL FEQA QUESTIONS
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

for idx, question in enumerate(questions):

    print("\n")
    print("=" * 60)
    print(f"QUESTION {idx+1}")
    print("=" * 60)

    print("\nQuestion:")
    print(question)



    # ========================================================
    # ANSWER FROM SOURCE DOCUMENT
    # ========================================================

    source_answer = qa_pipeline(
        question=question,
        context=legal_document
    )

    source_text = source_answer['answer']

    source_text = source_text.strip().lower()



    # ========================================================
    # ANSWER FROM GENERATED SUMMARY
    # ========================================================

    summary_answer = qa_pipeline(
        question=question,
        context=generated_summary
    )

    summary_text = summary_answer['answer']

    summary_text = summary_text.strip().lower()



    print("\nAnswer from SOURCE:")
    print(source_text)

    print("\nAnswer from SUMMARY:")
    print(summary_text)



    # ========================================================
    # SEMANTIC SIMILARITY USING InLegalBERT
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

    if similarity_score >= 0.90:

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

feqa_score = supported_questions / total_questions

hallucination_rate = (
    1 - feqa_score
) * 100



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
# DIRECT FINAL HALLUCINATION SCORE
# ============================================================

print("\n")
print("=" * 60)
print("FINAL HALLUCINATION SCORE")
print("=" * 60)

print(f"{round(hallucination_rate, 2)}%")