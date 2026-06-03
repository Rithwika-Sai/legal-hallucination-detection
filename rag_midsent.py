# =========================================================
# FAST MISTRAL RAG + MIDDLE HALLUCINATION DETECTION
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers==4.41.2
# pip install sentence-transformers==2.7.0
# pip install faiss-cpu
# pip install accelerate
# pip install spacy
# python -m spacy download en_core_web_sm

# =========================================================
# IMPORTS
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download("punkt")

import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer

import faiss

from transformers import pipeline

import torch

# =========================================================
# LOAD DOCUMENT
# =========================================================

with open(
    "data/test.json",
    "r",
    encoding="utf-8"
) as f:

    document = f.read()

# =========================================================
# CLEAN DOCUMENT
# =========================================================

document = document.replace("\n", " ")

document = " ".join(document.split())

# SHORTEN FOR FAST EXECUTION
document = document[:1500]

# =========================================================
# CHUNKING
# =========================================================

chunk_size = 300

chunks = [

    document[i:i+chunk_size]

    for i in range(0, len(document), chunk_size)
]

print(f"\nTotal Chunks: {len(chunks)}")

# =========================================================
# EMBEDDING MODEL
# =========================================================

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================================================
# CREATE EMBEDDINGS
# =========================================================

chunk_embeddings = embed_model.encode(

    chunks,

    convert_to_numpy=True
)

# =========================================================
# FAISS INDEX
# =========================================================

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(chunk_embeddings)

print("\nFAISS Index Ready")

# =========================================================
# LOAD MISTRAL
# =========================================================

print("\nLoading Mistral...\n")

mistral = pipeline(

    "text-generation",

    model="mistralai/Mistral-7B-Instruct-v0.2",

    torch_dtype=torch.float16,

    device_map="auto"
)

# =========================================================
# LOAD NLI MODEL
# =========================================================

print("\nLoading NLI Model...\n")

nli_model = pipeline(

    "text-classification",

    model="facebook/bart-large-mnli"
)

# =========================================================
# RETRIEVAL FUNCTION
# =========================================================

def retrieve(query, top_k=1):

    query_embedding = embed_model.encode(

        [query],

        convert_to_numpy=True
    )

    distances, indices = index.search(

        query_embedding,

        top_k
    )

    retrieved = [

        chunks[idx]

        for idx in indices[0]
    ]

    return retrieved

# =========================================================
# POSITION-AWARE RETRIEVAL
# =========================================================

base_chunks = retrieve(

    "legal judgment summary",

    top_k=1
)

middle_chunks = retrieve(

    "important legal reasoning",

    top_k=1
)

context = " ".join(

    base_chunks +

    middle_chunks
)

print("\nRetrieved Context Ready")

# =========================================================
# GENERATION PROMPT
# =========================================================

prompt = f"""
Summarize the following legal judgment
in exactly 3 factual sentences.

Document:
{context}

Summary:
"""

# =========================================================
# GENERATE SUMMARY
# =========================================================

output = mistral(

    prompt,

    max_new_tokens=40,

    do_sample=False,

    temperature=0.1
)

summary = output[0]["generated_text"]

summary = summary.replace(prompt, "")

print("\n")
print("="*80)
print("GENERATED SUMMARY")
print("="*80)

print(summary)

# =========================================================
# TOKENIZE SENTENCES
# =========================================================

sentences = sent_tokenize(summary)

if len(sentences) < 3:

    print("\nSummary too short.")
    exit()

target_sentences = sentences[:3]

# =========================================================
# HALLUCINATION DETECTION
# =========================================================

results = []

for idx, sent in enumerate(

    target_sentences,

    start=1
):

    nli_input = f"""
    premise: {document}

    hypothesis: {sent}
    """

    result = nli_model(nli_input)[0]

    label = result["label"]

    score = result["score"]

    hallucinated = (

        label != "ENTAILMENT"
    )

    # =====================================================
    # MIDDLE SENTENCE REGENERATION
    # =====================================================

    regenerated = False

    if idx == 2 and hallucinated:

        print("\nMiddle sentence hallucination detected.")
        print("Regenerating...\n")

        regen_prompt = f"""
        Rewrite factually:

        Sentence:
        {sent}

        Evidence:
        {context}

        Corrected Sentence:
        """

        regen_output = mistral(

            regen_prompt,

            max_new_tokens=30,

            do_sample=False,

            temperature=0.1
        )

        corrected = regen_output[0]["generated_text"]

        corrected = corrected.replace(
            regen_prompt,
            ""
        )

        # Re-verify
        verify_input = f"""
        premise: {document}

        hypothesis: {corrected}
        """

        verify_result = nli_model(
            verify_input
        )[0]

        if verify_result["label"] == "ENTAILMENT":

            sent = corrected

            label = verify_result["label"]

            score = verify_result["score"]

            hallucinated = False

            regenerated = True

    results.append({

        "Sentence Position": idx,

        "Sentence": sent,

        "NLI Label": label,

        "Confidence": round(score, 3),

        "Hallucinated": hallucinated,

        "Regenerated": regenerated
    })

# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("="*90)
print("FINAL RESULTS")
print("="*90)

print(results_df)

# =========================================================
# HALLUCINATION RATE
# =========================================================

hall_count = results_df["Hallucinated"].sum()

hall_rate = (

    hall_count / len(results_df)

) * 100

print("\n")
print("="*50)

print(f"Final Hallucination Rate: {hall_rate:.2f}%")

print("="*50)

# =========================================================
# SAVE CSV
# =========================================================

results_df.to_csv(

    "mistral_middle_rag_results.csv",

    index=False
)

print("\nSaved:")
print("mistral_middle_rag_results.csv")

print("\nDONE SUCCESSFULLY")