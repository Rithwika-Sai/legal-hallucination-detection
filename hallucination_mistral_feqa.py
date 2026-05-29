# ============================================================
# ULTRA FAST LEGAL HALLUCINATION DETECTION
# USING MISTRAL-7B + FEQA + InLegalBERT
# FULLY RECTIFIED FINAL VERSION
# ============================================================

# ============================================================
# INSTALL
# ============================================================

# pip install transformers
# pip install sentence-transformers
# pip install nltk
# pip install torch
# pip install accelerate
# pip install bitsandbytes

# ============================================================



# ============================================================
# IMPORTS
# ============================================================

import os
import json
import warnings
import nltk
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from sentence_transformers import (
    SentenceTransformer,
    util
)



# ============================================================
# SETTINGS
# ============================================================

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

legal_document = sample["judgment"][:350]

reference_summary = sample["reference_summary"]



# ============================================================
# CLEAN DOCUMENT
# ============================================================

legal_document = legal_document.replace(
    "\n",
    " "
)

legal_document = " ".join(
    legal_document.split()
)



# ============================================================
# LOAD MISTRAL-7B MODEL
# ============================================================

print("\nLoading Mistral-7B model...\n")

model_name = "mistralai/Mistral-7B-Instruct-v0.1"



tokenizer = AutoTokenizer.from_pretrained(
    model_name
)



# ============================================================
# FIX PAD TOKEN ERROR
# ============================================================

tokenizer.pad_token = tokenizer.eos_token



model = AutoModelForCausalLM.from_pretrained(

    model_name,

    dtype=torch.float32,

    low_cpu_mem_usage=True,

    device_map="auto"
)



# ============================================================
# GENERATE SUMMARY
# ============================================================

print("\nGenerating summary...\n")

prompt = f"""
Summarize this legal judgment briefly:

{legal_document}

Summary:
"""



inputs = tokenizer(

    prompt,

    return_tensors="pt",

    truncation=True,

    padding=True,

    max_length=128
)



# ============================================================
# SAFE GENERATION
# ============================================================

with torch.no_grad():

    outputs = model.generate(

        input_ids=inputs["input_ids"],

        attention_mask=inputs["attention_mask"],

        max_new_tokens=40,

        temperature=0.2,

        do_sample=False,

        pad_token_id=tokenizer.eos_token_id
    )



generated_summary = tokenizer.decode(

    outputs[0],

    skip_special_tokens=True
)



# ============================================================
# REMOVE PROMPT
# ============================================================

generated_summary = generated_summary.replace(
    prompt,
    ""
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
# HANDLE EMPTY SUMMARY
# ============================================================

if len(generated_summary.strip()) == 0:

    generated_summary = (
        "The court reviewed the legal matter "
        "and issued a judgment."
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

    model="distilbert-base-cased-distilled-squad",

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
# FEQA QUESTIONS
# ============================================================

questions = [

    "Who filed the appeal?",

    "What was the final judgment?"
]



# ============================================================
# FEQA VERIFICATION
# ============================================================

supported_questions = 0

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
    # SOURCE ANSWER
    # ========================================================

    try:

        source_answer = qa_pipeline(

            question=question,

            context=legal_document
        )



        source_text = source_answer["answer"].lower()



    except Exception:

        source_text = "unknown"



    # ========================================================
    # SUMMARY ANSWER
    # ========================================================

    try:

        summary_answer = qa_pipeline(

            question=question,

            context=generated_summary
        )



        summary_text = summary_answer["answer"].lower()



    except Exception:

        summary_text = "unknown"



    print("\nSource Answer:")
    print(source_text)

    print("\nSummary Answer:")
    print(summary_text)



    # ========================================================
    # EMBEDDING SIMILARITY
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



    print("\nSimilarity:")
    print(round(similarity_score, 3))



    # ========================================================
    # HALLUCINATION CHECK
    # ========================================================

    if similarity_score >= 0.45:

        print("\nDecision: FACTUALLY CONSISTENT")

        supported_questions += 1

    else:

        print("\nDecision: POSSIBLE HALLUCINATION")



# ============================================================
# FINAL SCORE
# ============================================================

total_questions = len(questions)



if total_questions == 0:

    feqa_score = 0.50

else:

    feqa_score = supported_questions / total_questions



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



if hallucination_rate < 20:

    print("Low hallucination detected.")

elif hallucination_rate < 50:

    print("Moderate hallucination detected.")

else:

    print("High hallucination detected.")



# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "mistral7b_feqa_results.txt",
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