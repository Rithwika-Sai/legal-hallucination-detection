# =====================================================
# HYBRID HALLUCINATION DETECTION
# USING MISTRAL-7B + InLegalBERT
# Entity Mismatch + Semantic Similarity
# =====================================================

# =====================================================
# INSTALL
# =====================================================

# pip install transformers
# pip install sentence-transformers
# pip install spacy
# pip install torch
# pip install accelerate
# pip install bitsandbytes
# pip install nltk

# python -m spacy download en_core_web_sm

# =====================================================
# SUPPRESS WARNINGS
# =====================================================

import warnings
warnings.filterwarnings("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from transformers import logging
logging.set_verbosity_error()

import logging as py_logging
py_logging.disable(py_logging.CRITICAL)

# =====================================================
# IMPORTS
# =====================================================

import torch
import spacy
import nltk

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from sentence_transformers import (
    SentenceTransformer,
    util
)

# =====================================================
# DOWNLOAD NLTK
# =====================================================

nltk.download("punkt", quiet=True)

# =====================================================
# LOAD SPACY
# =====================================================

print("\nLoading spaCy...\n")

nlp = spacy.load("en_core_web_sm")

# =====================================================
# LOAD InLegalBERT
# =====================================================

print("\nLoading InLegalBERT...\n")

embed_model = SentenceTransformer(
    "law-ai/InLegalBERT"
)

# =====================================================
# LOAD MISTRAL-7B
# =====================================================

print("\nLoading Mistral-7B...\n")

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(

    model_name,

    torch_dtype=torch.float32,

    low_cpu_mem_usage=True,

    device_map="auto"
)

# =====================================================
# READ SOURCE DOCUMENT
# =====================================================

with open(
    "data/test.json",
    "r",
    encoding="utf-8"
) as f:

    source = f.read()

# =====================================================
# CLEAN SOURCE
# =====================================================

source = source.replace("\n", " ")

source = " ".join(
    source.split()
)

# =====================================================
# LIMIT DOCUMENT SIZE
# =====================================================

source_document = source[:1000]

# =====================================================
# GENERATE SUMMARY USING MISTRAL
# =====================================================

print("\nGenerating Summary...\n")

prompt = f"""
Summarize this legal judgment briefly:

{source_document}

Summary:
"""

inputs = tokenizer(

    prompt,

    return_tensors="pt",

    truncation=True,

    max_length=512,

    padding=True
)

# =====================================================
# GENERATE OUTPUT
# =====================================================

with torch.no_grad():

    outputs = model.generate(

        input_ids=inputs["input_ids"],

        attention_mask=inputs["attention_mask"],

        max_new_tokens=120,

        do_sample=False,

        temperature=0.2,

        pad_token_id=tokenizer.eos_token_id
    )

# =====================================================
# DECODE SUMMARY
# =====================================================

generated_summary = tokenizer.decode(

    outputs[0],

    skip_special_tokens=True
)

# =====================================================
# REMOVE PROMPT
# =====================================================

generated_summary = generated_summary.replace(
    prompt,
    ""
)

# =====================================================
# CLEAN SUMMARY
# =====================================================

generated_summary = generated_summary.replace(
    "\n",
    " "
)

generated_summary = " ".join(
    generated_summary.split()
)

# =====================================================
# HANDLE EMPTY SUMMARY
# =====================================================

if len(generated_summary.strip()) == 0:

    generated_summary = (
        "The court reviewed the legal matter "
        "and delivered the judgment."
    )

# =====================================================
# PRINT SUMMARY
# =====================================================

print("\n===== GENERATED SUMMARY =====\n")

print(generated_summary)

# =====================================================
# ENTITY EXTRACTION
# =====================================================

print("\nExtracting entities...\n")

source_doc = nlp(source[:5000])

summary_doc = nlp(generated_summary)

# =====================================================
# GET ENTITIES
# =====================================================

source_entities = set(

    ent.text.lower()

    for ent in source_doc.ents
)

summary_entities = set(

    ent.text.lower()

    for ent in summary_doc.ents
)

# =====================================================
# FIND UNSUPPORTED ENTITIES
# =====================================================

unsupported_entities = []

for ent in summary_entities:

    if ent not in source_entities:

        unsupported_entities.append(ent)

# =====================================================
# ENTITY HALLUCINATION SCORE
# =====================================================

total_entities = len(summary_entities)

if total_entities == 0:

    entity_score = 0

else:

    entity_score = (

        len(unsupported_entities)

        / total_entities

    ) * 100

# =====================================================
# SEMANTIC SIMILARITY
# =====================================================

print("\nComputing semantic similarity...\n")

source_embedding = embed_model.encode(

    source[:3000],

    convert_to_tensor=True
)

summary_embedding = embed_model.encode(

    generated_summary,

    convert_to_tensor=True
)

similarity = util.cos_sim(

    source_embedding,

    summary_embedding
)

semantic_similarity = similarity.item()

# =====================================================
# SEMANTIC HALLUCINATION SCORE
# =====================================================

semantic_score = (

    1 - semantic_similarity

) * 100

# =====================================================
# FINAL HYBRID HALLUCINATION SCORE
# =====================================================

hallucination_score = (

    0.5 * entity_score

    +

    0.5 * semantic_score
)

# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n")
print("=" * 60)

print(
    f"\nEntity Hallucination Score: "
    f"{entity_score:.2f}%"
)

print(
    f"\nSemantic Hallucination Score: "
    f"{semantic_score:.2f}%"
)

print(
    f"\nFinal Hybrid Hallucination Score: "
    f"{hallucination_score:.2f}%"
)

print("\nUnsupported Entities:\n")

if len(unsupported_entities) == 0:

    print("No unsupported entities found.")

else:

    for ent in unsupported_entities:

        print("-", ent)

# =====================================================
# FINAL DECISION
# =====================================================

print("\n")

if hallucination_score > 30:

    print("Hallucination Detected")

else:

    print("Summary is Mostly Faithful")

print("\n")
print("=" * 60)
print("\nDONE SUCCESSFULLY\n")











