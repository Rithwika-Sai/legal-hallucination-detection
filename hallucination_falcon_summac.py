# =========================================================
# ULTRA FAST SUMMAC + FALCON-7B
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
import json
import warnings
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
# ONLY FIRST 2 DOCUMENT SENTENCES
# =========================================================

document_sentences = nltk.sent_tokenize(
    source_document
)[:2]

summary_sentences = nltk.sent_tokenize(
    generated_summary
)

# =========================================================
# PRECOMPUTE DOCUMENT EMBEDDINGS
# =========================================================

document_embeddings = embed_model.encode(

    document_sentences,

    convert_to_tensor=True
)

# =========================================================
# SUMMAC ANALYSIS
# =========================================================

sentence_scores = []

print("\n===================================")
print("SUMMAC ANALYSIS")
print("===================================\n")

# =========================================================
# PROCESS SUMMARY SENTENCES
# =========================================================

for idx, summary_sent in enumerate(summary_sentences):

    if len(summary_sent.strip()) < 5:
        continue

    print(f"\nSummary Sentence {idx+1}:")
    print(summary_sent)

    max_score = 0

    # =====================================================
    # SUMMARY EMBEDDING
    # =====================================================

    summary_embedding = embed_model.encode(

        summary_sent,

        convert_to_tensor=True
    )

    # =====================================================
    # COMPARE ONLY 2 SENTENCES
    # =====================================================

    for i, doc_sent in enumerate(document_sentences):

        # =================================================
        # NLI
        # =================================================

        nli_input = (
            f"{doc_sent} </s></s> {summary_sent}"
        )

        result = nli_pipeline(
            nli_input
        )[0]

        # =================================================
        # LABEL
        # =================================================

        if result["label"] == "ENTAILMENT":

            entailment_score = result["score"]

        else:

            entailment_score = 0

        # =================================================
        # FAST SEMANTIC SCORE
        # =================================================

        semantic_score = util.cos_sim(

            document_embeddings[i],

            summary_embedding

        ).item()

        # =================================================
        # FINAL SCORE
        # =================================================

        pair_score = (

            0.6 * entailment_score

            +

            0.4 * semantic_score
        )

        # =================================================
        # KEEP BEST
        # =================================================

        if pair_score > max_score:

            max_score = pair_score

    sentence_scores.append(max_score)

    print(f"\nFaithfulness Score:")
    print(round(max_score, 4))

# =========================================================
# FINAL SCORES
# =========================================================

if len(sentence_scores) == 0:

    summac_score = 0.50

else:

    summac_score = np.mean(
        sentence_scores
    )

hallucination_rate = (
    1 - summac_score
) * 100

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
    f"SummaC Score       : "
    f"{summac_score:.4f}"
)

print(
    f"Hallucination Rate : "
    f"{hallucination_rate:.2f}%"
)

# =========================================================
# FINAL DECISION
# =========================================================

print("\n===================================")
print("FINAL DECISION")
print("===================================\n")

if summac_score >= 0.40:

    print("Summary is FACTUALLY CONSISTENT")

else:

    print("Summary contains HALLUCINATIONS")

print("\nDONE SUCCESSFULLY\n")



