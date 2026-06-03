# =========================================================
# MISTRAL RAG + SENTENCE LEVEL HALLUCINATION DETECTION
# =========================================================
# Detect hallucination for:
# - 1st sentence
# - 2nd sentence
# - 3rd sentence
#
# USING:
# - Mistral-7B-Instruct
# - FAISS RAG
# - SummaC
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install accelerate
# pip install bitsandbytes
# pip install sentence-transformers
# pip install faiss-cpu
# pip install nltk
# pip install summac
# pip install pandas
# pip install torch

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

from summac.model_summac import SummaCZS

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

# =========================================================
# SHORTEN FOR FAST EXECUTION
# =========================================================

document = document[:3000]

# =========================================================
# CHUNK DOCUMENT
# =========================================================

chunk_size = 400

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
# EMBEDDINGS
# =========================================================

chunk_embeddings = embed_model.encode(

    chunks,

    convert_to_numpy=True
)

# =========================================================
# CREATE FAISS INDEX
# =========================================================

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(chunk_embeddings)

print("\nFAISS Index Ready")

# =========================================================
# LOAD MISTRAL
# =========================================================

print("\nLoading Mistral...\n")

generator = pipeline(

    "text-generation",

    model="mistralai/Mistral-7B-Instruct-v0.1",

    device_map="auto"
)

# =========================================================
# LOAD SUMMAC
# =========================================================

print("\nLoading SummaC...\n")

summac_model = SummaCZS(

    granularity="sentence",

    model_name="vitc"
)

# =========================================================
# QUERY
# =========================================================

query = "Summarize the legal judgment"

# =========================================================
# QUERY EMBEDDING
# =========================================================

query_embedding = embed_model.encode(

    [query],

    convert_to_numpy=True
)

# =========================================================
# RETRIEVE TOP CHUNKS
# =========================================================

top_k = 3

distances, indices = index.search(

    query_embedding,

    top_k
)

retrieved_chunks = [

    chunks[idx]

    for idx in indices[0]
]

# =========================================================
# BUILD CONTEXT
# =========================================================

context = " ".join(retrieved_chunks)

print("\nRetrieved Context Ready")

# =========================================================
# MISTRAL PROMPT
# =========================================================

prompt = f"""
You are a legal summarization assistant.

Summarize the following legal judgment
in 3 concise factual sentences.

Document:
{context}

Summary:
"""

# =========================================================
# GENERATE SUMMARY
# =========================================================

output = generator(

    prompt,

    max_new_tokens=120,

    do_sample=False,

    temperature=0.2
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

# =========================================================
# ENSURE 3 SENTENCES
# =========================================================

if len(sentences) < 3:

    print("\nSummary too short.")
    exit()

# =========================================================
# TAKE FIRST 3 SENTENCES
# =========================================================

first_sentence = sentences[0]

second_sentence = sentences[1]

third_sentence = sentences[2]

# =========================================================
# HALLUCINATION CHECK FUNCTION
# =========================================================

threshold = 0.5

def hallucination_check(sentence):

    score = summac_model.score(

        [document],

        [sentence]

    )["scores"][0]

    hallucinated = score < threshold

    return score, hallucinated

# =========================================================
# CHECK SENTENCES
# =========================================================

results = []

for idx, sent in enumerate(

    [

        first_sentence,

        second_sentence,

        third_sentence

    ],

    start=1
):

    score, hall = hallucination_check(sent)

    results.append({

        "Sentence Position": idx,

        "Sentence": sent,

        "SummaC Score": round(score, 3),

        "Hallucinated": hall
    })

# =========================================================
# RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("="*80)
print("SENTENCE LEVEL HALLUCINATION RESULTS")
print("="*80)

print(results_df)

# =========================================================
# TOTAL HALLUCINATION RATE
# =========================================================

hall_count = results_df["Hallucinated"].sum()

hall_rate = (

    hall_count / len(results_df)

) * 100

print("\n")
print("="*50)

print(f"Overall Hallucination Rate: {hall_rate:.2f}%")

print("="*50)

# =========================================================
# POSITION-WISE %
# =========================================================

for _, row in results_df.iterrows():

    position = row["Sentence Position"]

    if row["Hallucinated"]:

        percent = 100

    else:

        percent = 0

    print(

        f"Sentence {position} Hallucination %: {percent}%"
    )

# =========================================================
# SAVE CSV
# =========================================================

results_df.to_csv(

    "mistral_sentence_hallucination.csv",

    index=False
)

print("\nSaved:")
print("mistral_sentence_hallucination.csv")

print("\nDONE SUCCESSFULLY")