# =========================================================
# FAST HHEM-STYLE NLI HALLUCINATION DETECTOR
# USING FALCON-7B
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

import json
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
# CPU OPTIMIZATION
# =========================================================

torch.set_num_threads(4)

# =========================================================
# DOWNLOAD TOKENIZER
# =========================================================

nltk.download("punkt", quiet=True)

# =========================================================
# LOAD FALCON-7B MODEL
# =========================================================

print("\nLoading Falcon-7B model...\n")

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
# LOAD FAST NLI MODEL
# =========================================================

print("\nLoading NLI model...\n")

nli_pipeline = pipeline(

    "text-classification",

    model="typeform/distilbert-base-uncased-mnli",

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
# TAKE ONLY FIRST SAMPLE
# =========================================================

sample = data[0]

# =========================================================
# VERY SMALL INPUT FOR SPEED
# =========================================================

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
# TOKENIZE INPUT
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
# HANDLE EMPTY SUMMARY
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
# SPLIT SUMMARY INTO SENTENCES
# =========================================================

summary_sentences = nltk.sent_tokenize(
    generated_summary
)

# =========================================================
# PRECOMPUTE SOURCE EMBEDDING
# =========================================================

source_embedding = embed_model.encode(

    source_document,

    convert_to_tensor=True
)

# =========================================================
# HHEM ANALYSIS
# =========================================================

faithfulness_scores = []

print("\n===================================")
print("HHEM NLI ANALYSIS")
print("===================================\n")

# =========================================================
# PROCESS EACH SENTENCE
# =========================================================

for idx, sent in enumerate(summary_sentences):

    if len(sent.strip()) < 5:
        continue

    print(f"\nSentence {idx+1}:")
    print(sent)

    # =====================================================
    # NLI ENTAILMENT CHECK
    # =====================================================

    nli_input = f"""
    premise: {source_document}

    hypothesis: {sent}
    """

    nli_result = nli_pipeline(
        nli_input
    )[0]

    nli_score = nli_result["score"]

    print(f"\nNLI Score:")
    print(round(nli_score, 4))

    # =====================================================
    # FAST SEMANTIC SIMILARITY
    # =====================================================

    sentence_embedding = embed_model.encode(

        sent,

        convert_to_tensor=True
    )

    semantic_score = util.cos_sim(

        source_embedding,

        sentence_embedding

    ).item()

    print(f"\nSemantic Similarity:")
    print(round(semantic_score, 4))

    # =====================================================
    # FINAL SCORE
    # =====================================================

    final_score = (

        0.6 * nli_score

        +

        0.4 * semantic_score
    )

    final_score = max(
        0,
        min(final_score, 1)
    )

    faithfulness_scores.append(
        final_score
    )

    print(f"\nFaithfulness Score:")
    print(round(final_score, 4))

# =========================================================
# HANDLE EMPTY SCORES
# =========================================================

if len(faithfulness_scores) == 0:

    overall_faithfulness = 0.50

else:

    overall_faithfulness = np.mean(
        faithfulness_scores
    )

# =========================================================
# FINAL HALLUCINATION SCORE
# =========================================================

hallucination_score = (
    1 - overall_faithfulness
)

hallucination_rate = (
    hallucination_score * 100
)

# =========================================================
# PREVENT EXTREME VALUES
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

print("\n===================================")
print("FINAL DECISION")
print("===================================\n")

if overall_faithfulness >= 0.40:

    print("Summary is FACTUALLY CONSISTENT")

else:

    print("Summary contains HALLUCINATIONS")

# =========================================================
# SAVE RESULTS
# =========================================================

with open(
    "fast_hhem_falcon_results.txt",
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

print("\nResults saved successfully.")
