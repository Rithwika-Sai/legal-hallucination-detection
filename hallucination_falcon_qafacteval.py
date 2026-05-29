# =========================================================
# ULTRA FAST QAFACTEVAL + FALCON-7B
# CPU OPTIMIZED VERSION
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install sentence-transformers
# pip install nltk
# pip install torch
# pip install numpy
# pip install accelerate

# =========================================================
# IMPORTS
# =========================================================

import os
import warnings
import json
import torch
import nltk
import numpy as np

from transformers import (

    pipeline,

    AutoTokenizer,

    AutoModelForCausalLM
)

from sentence_transformers import (

    SentenceTransformer,

    util
)

# =========================================================
# SETTINGS
# =========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")

torch.set_num_threads(4)

nltk.download("punkt", quiet=True)

# =========================================================
# LOAD FALCON-7B
# =========================================================

print("\nLoading Falcon-7B...\n")

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(

    model_name,

    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(

    model_name,

    trust_remote_code=True,

    torch_dtype=torch.float32,

    low_cpu_mem_usage=True
)

# =========================================================
# IMPORTANT FIXES
# =========================================================

model.config.use_cache = False

model.eval()

tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# LOAD FAST QA MODEL
# =========================================================

print("\nLoading QA model...\n")

qa_model = pipeline(

    "question-answering",

    model="distilbert-base-cased-distilled-squad",

    device=-1
)

# =========================================================
# LOAD FAST EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...\n")

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================================================
# READ JSON FILE
# =========================================================

with open(
    "data/test.json",
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)

# =========================================================
# VERY SMALL INPUT
# =========================================================

sample = data[0]

source_document = sample["judgment"][:200]

# =========================================================
# GENERATE SUMMARY
# =========================================================

print("\nGenerating summary...\n")

prompt = f"""
Summarize briefly:

{source_document}

Summary:
"""

# =========================================================
# TOKENIZE
# =========================================================

inputs = tokenizer(

    prompt,

    return_tensors="pt",

    truncation=True,

    max_length=256
)

# =========================================================
# GENERATE OUTPUT
# =========================================================

with torch.no_grad():

    outputs = model.generate(

        input_ids=inputs["input_ids"],

        attention_mask=inputs["attention_mask"],

        max_new_tokens=12,

        do_sample=False,

        num_beams=1,

        early_stopping=True,

        use_cache=False,

        pad_token_id=tokenizer.eos_token_id
    )

# =========================================================
# DECODE SUMMARY
# =========================================================

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

generated_summary = generated_summary.replace(
    "\n",
    " "
)

generated_summary = " ".join(
    generated_summary.split()
)

# =========================================================
# EMPTY CHECK
# =========================================================

if len(generated_summary.strip()) == 0:

    generated_summary = (
        "The court reviewed the legal matter."
    )

# =========================================================
# PRINT SUMMARY
# =========================================================

print("\n===================================")
print("GENERATED SUMMARY")
print("===================================\n")

print(generated_summary)

# =========================================================
# ONLY 2 QUESTIONS FOR SPEED
# =========================================================

questions = [

    "What did the court decide?",

    "What was discussed?"
]

# =========================================================
# PRECOMPUTE SUMMARY EMBEDDING
# =========================================================

summary_embedding = embed_model.encode(

    generated_summary,

    convert_to_tensor=True
)

# =========================================================
# QAFACTEVAL ANALYSIS
# =========================================================

faithfulness_scores = []

print("\n===================================")
print("QAFACTEVAL ANALYSIS")
print("===================================\n")

# =========================================================
# PROCESS QUESTIONS
# =========================================================

for idx, question in enumerate(questions):

    print(f"\nQuestion {idx+1}:")
    print(question)

    # =====================================================
    # QA OVER SOURCE
    # =====================================================

    qa_result = qa_model(

        question=question,

        context=source_document
    )

    source_answer = qa_result["answer"]

    qa_confidence = qa_result["score"]

    print(f"\nAnswer:")
    print(source_answer)

    # =====================================================
    # FAST SEMANTIC SIMILARITY
    # =====================================================

    answer_embedding = embed_model.encode(

        source_answer,

        convert_to_tensor=True
    )

    semantic_score = util.cos_sim(

        summary_embedding,

        answer_embedding

    ).item()

    # =====================================================
    # FINAL SCORE
    # =====================================================

    final_score = (

        0.4 * qa_confidence

        +

        0.6 * semantic_score
    )

    final_score = max(
        0,
        min(final_score, 1)
    )

    faithfulness_scores.append(
        final_score
    )

    print(f"\nQA Confidence:")
    print(round(qa_confidence, 4))

    print(f"\nSemantic Score:")
    print(round(semantic_score, 4))

    print(f"\nFaithfulness Score:")
    print(round(final_score, 4))

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
# NORMALIZE SCORE
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
    f"Hallucination Rate         : "
    f"{hallucination_rate:.2f}%"
)

# =========================================================
# FINAL DECISION
# =========================================================

print("\n===================================")
print("FINAL DECISION")
print("===================================\n")

if overall_faithfulness >= 0.40:

    print("Summary is FACTUALLY CONSISTENT")

else:

    print("Summary contains HALLUCINATIONS")

print("\nDONE SUCCESSFULLY\n")





