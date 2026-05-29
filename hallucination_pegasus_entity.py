# =====================================================
# HYBRID HALLUCINATION DETECTION
# Entity Mismatch + Semantic Similarity
# =====================================================

# INSTALL:
# pip install spacy sentence-transformers
# python -m spacy download en_core_web_sm

# =====================================================
# SUPPRESS WARNINGS
# =====================================================

# =====================================================
# IGNORE ALL WARNINGS & LOGS
# =====================================================

import warnings
warnings.filterwarnings("ignore")

import os

# TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Transformers warnings
from transformers import logging
logging.set_verbosity_error()

# Python warnings
import logging as py_logging
py_logging.disable(py_logging.CRITICAL)

# =====================================================
# IMPORTS
# =====================================================

import spacy

from sentence_transformers import (
    SentenceTransformer,
    util
)

# =====================================================
# LOAD MODELS
# =====================================================

nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer(
    'law-ai/InLegalBERT'
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
# READ GENERATED SUMMARY
# =====================================================

with open(
    "outputs/pegasus_output.txt",
    "r",
    encoding="utf-8"
) as f:

    summary = f.read()

# =====================================================
# CLEAN SUMMARY
# =====================================================

summary = summary.replace("\\n", " ")

# =====================================================
# PRINT SUMMARY
# =====================================================

print("\n===== SUMMARY =====\n")

print(summary)

# =====================================================
# ENTITY EXTRACTION
# =====================================================

source_doc = nlp(source[:50000])

summary_doc = nlp(summary)

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

source_embedding = embed_model.encode(
    source[:5000],
    convert_to_tensor=True
)

summary_embedding = embed_model.encode(
    summary,
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
# FINAL HYBRID SCORE
# =====================================================

hallucination_score = (

    0.5 * entity_score

    +

    0.5 * semantic_score
)

# =====================================================
# FINAL OUTPUT
# =====================================================

print(
    f"\nHallucination Score: "
    f"{hallucination_score:.2f}%"
)

if hallucination_score > 0:

    print("Hallucination Detected")

else:

    print("No Hallucination")