# =========================================================
# FINAL HHEM-STYLE NLI HALLUCINATION DETECTOR
# USING TinyLlama + InLegalBERT
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install sentence-transformers
# pip install nltk
# pip install torch
# pip install accelerate
# pip install sentencepiece

# =========================================================
# IMPORTS
# =========================================================

import os
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
# HIDE WARNINGS
# =========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# =========================================================
# DOWNLOAD TOKENIZER
# =========================================================

nltk.download("punkt", quiet=True)

# =========================================================
# LOAD TinyLlama MODEL
# =========================================================

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

# =========================================================
# LOAD NLI MODEL
# =========================================================

print("\nLoading NLI model...\n")

nli_pipeline = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    device=-1
)

# =========================================================
# LOAD InLegalBERT
# =========================================================

print("\nLoading InLegalBERT...\n")

embed_model = SentenceTransformer(
    "law-ai/InLegalBERT"
)

# =========================================================
# SOURCE LEGAL DOCUMENT
# =========================================================

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

# =========================================================
# GENERATE SUMMARY USING TinyLlama
# =========================================================

print("\nGenerating summary...\n")

prompt = f"""
Summarize the following legal judgment clearly and concisely:

{source_document}

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

# =========================================================
# CLEAN SUMMARY
# =========================================================

generated_summary = generated_summary.replace(
    prompt,
    ""
).strip()

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
# HHEM-STYLE ANALYSIS
# =========================================================

faithfulness_scores = []

print("\n===================================")
print("HHEM NLI ANALYSIS")
print("===================================\n")

# =========================================================
# PROCESS EACH SUMMARY SENTENCE
# =========================================================

for idx, sent in enumerate(summary_sentences):

    # Skip very small sentences
    if len(sent.strip()) < 10:
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

    nli_label = nli_result['label']
    nli_score = nli_result['score']

    print(f"\nNLI Label:")
    print(nli_label)

    print(f"\nNLI Score:")
    print(round(nli_score, 4))

    # =====================================================
    # SEMANTIC SIMILARITY
    # =====================================================

    source_embedding = embed_model.encode(
        source_document,
        convert_to_tensor=True
    )

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
    # HHEM-STYLE FINAL SCORE
    # =====================================================

    # Weight:
    # 70% NLI
    # 30% Semantic Similarity

    final_score = (
        0.7 * nli_score
        +
        0.3 * semantic_score
    )

    # Clamp score
    final_score = max(0, min(final_score, 1))

    faithfulness_scores.append(final_score)

    print(f"\nFaithfulness Score:")
    print(round(final_score, 4))

    # =====================================================
    # DECISION
    # =====================================================

    THRESHOLD = 0.75

    if final_score >= THRESHOLD:

        print("\nDecision:")
        print("FACTUALLY CONSISTENT")

    else:

        print("\nDecision:")
        print("POSSIBLE HALLUCINATION")

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

if overall_faithfulness >= 0.75:

    print("Summary is FACTUALLY CONSISTENT")

else:

    print("Summary contains HALLUCINATIONS")

# =========================================================
# DIRECT FINAL HALLUCINATION SCORE
# =========================================================

print("\n===================================")
print("FINAL HALLUCINATION SCORE")
print("===================================\n")

print(f"{hallucination_rate:.2f}%")

# =========================================================
# SAVE RESULTS
# =========================================================

with open(
    "hhem_tinyllama_results.txt",
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