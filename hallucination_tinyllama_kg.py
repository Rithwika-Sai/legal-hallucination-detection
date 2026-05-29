# =========================================================
# KNOWLEDGE GRAPH (KG) HALLUCINATION DETECTION
# =========================================================
# Features:
# ✅ Knowledge Graph based
# ✅ No 0% issue
# ✅ No 100% issue
# ✅ Decimal hallucination scores
# ✅ Handles legal summaries
# ✅ CPU friendly
# ✅ Prints ONLY summary + hall score
# =========================================================

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

import nltk
import spacy
import networkx as nx

from sentence_transformers import (
    SentenceTransformer,
    util
)

# =========================================================
# DOWNLOAD TOKENIZER
# =========================================================

nltk.download('punkt', quiet=True)

# =========================================================
# LOAD MODELS
# =========================================================

# spaCy
nlp = spacy.load("en_core_web_sm")

# Prevent max length crash
nlp.max_length = 3000000

# Embedding model
embed_model = SentenceTransformer(
    'law-ai/InLegalBERT'
)

# =========================================================
# READ SOURCE DOCUMENT
# =========================================================

with open("data/test.json", "r", encoding="utf-8") as f:
    source = f.read()

# =========================================================
# READ GENERATED SUMMARY
# =========================================================

with open("outputs/tinyllama_output.txt", "r", encoding="utf-8") as f:
    summary = f.read()

# =========================================================
# LIMIT HUGE LEGAL DOC
# =========================================================

source = source[:50000]

# =========================================================
# SPLIT INTO SENTENCES
# =========================================================

source_sentences = nltk.sent_tokenize(source)

summary_sentences = nltk.sent_tokenize(summary)

# =========================================================
# BUILD SOURCE KNOWLEDGE GRAPH
# =========================================================

source_graph = nx.Graph()

source_entities = []

# ---------------------------------------------------------
# EXTRACT ENTITIES FROM SOURCE
# ---------------------------------------------------------

for sent in source_sentences:

    doc = nlp(sent)

    ents = []

    for ent in doc.ents:

        entity = ent.text.lower().strip()

        ents.append(entity)

        source_entities.append(entity)

        source_graph.add_node(entity)

    # -----------------------------------------------------
    # CONNECT ENTITIES IN SAME SENTENCE
    # -----------------------------------------------------

    for i in range(len(ents)):

        for j in range(i + 1, len(ents)):

            source_graph.add_edge(
                ents[i],
                ents[j]
            )

# Remove duplicates
source_entities = list(set(source_entities))

# =========================================================
# SOURCE ENTITY EMBEDDINGS
# =========================================================

source_entity_embeddings = embed_model.encode(
    source_entities,
    convert_to_tensor=True
)

# =========================================================
# HALLUCINATION SCORE
# =========================================================

hallucination_total = 0

# =========================================================
# PROCESS SUMMARY SENTENCES
# =========================================================

for sent in summary_sentences:

    # Skip tiny sentences
    if len(sent.strip()) < 15:
        continue

    sentence_score = 0

    doc = nlp(sent)

    summary_entities = []

    # -----------------------------------------------------
    # EXTRACT SUMMARY ENTITIES
    # -----------------------------------------------------

    for ent in doc.ents:

        entity = ent.text.lower().strip()

        summary_entities.append(entity)

    # -----------------------------------------------------
    # NO ENTITIES
    # -----------------------------------------------------

    if len(summary_entities) == 0:

        sentence_score += 0.15

    # -----------------------------------------------------
    # ENTITY MATCHING
    # -----------------------------------------------------

    for summary_entity in summary_entities:

        summary_embedding = embed_model.encode(
            summary_entity,
            convert_to_tensor=True
        )

        similarities = util.cos_sim(
            summary_embedding,
            source_entity_embeddings
        )

        max_similarity = similarities.max().item()

        # -------------------------------------------------
        # DYNAMIC KG SCORE
        # -------------------------------------------------

        # Strong mismatch
        if max_similarity < 0.30:

            sentence_score += (
                (1 - max_similarity) * 0.8
            )

        # Partial mismatch
        elif max_similarity < 0.55:

            sentence_score += (
                (1 - max_similarity) * 0.4
            )

    # -----------------------------------------------------
    # RELATION CHECK
    # -----------------------------------------------------

    if len(summary_entities) >= 2:

        connected_pairs = 0
        total_pairs = 0

        for i in range(len(summary_entities)):

            for j in range(i + 1, len(summary_entities)):

                total_pairs += 1

                e1 = summary_entities[i]
                e2 = summary_entities[j]

                if source_graph.has_edge(e1, e2):

                    connected_pairs += 1

        # -------------------------------------------------
        # RELATION SCORE
        # -------------------------------------------------

        if total_pairs > 0:

            relation_score = (
                connected_pairs / total_pairs
            )

            # Penalize missing relations
            sentence_score += (
                (1 - relation_score) * 0.3
            )

    # -----------------------------------------------------
    # NORMALIZE
    # -----------------------------------------------------

    sentence_score = min(sentence_score, 1.0)

    hallucination_total += sentence_score

# =========================================================
# FINAL HALLUCINATION RATE
# =========================================================

total_sentences = len(summary_sentences)

hallucination_rate = (
    hallucination_total / total_sentences
) * 100

# =========================================================
# PRINT OUTPUT
# =========================================================

print("\n==============================")
print("SUMMARY")
print("==============================\n")

print(summary)

print("\n==============================")
print(
    f"HALLUCINATION RATE: "
    f"{hallucination_rate:.2f}%"
)
print("==============================")