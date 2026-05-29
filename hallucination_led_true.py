# =========================================================
# FAST TRUE NLI + InLegalBERT + LED
# FULL FIXED VERSION
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install sentence-transformers
# pip install nltk
# pip install torch
# pip install sentencepiece

# =========================================================
# IMPORTS
# =========================================================

import os
import warnings
import nltk
import numpy as np

from transformers import (
    LEDTokenizer,
    LEDForConditionalGeneration,
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

nltk.download("punkt", quiet=True)

# =========================================================
# LOAD LED MODEL
# =========================================================

print("\nLoading LED model...\n")

summarizer_name = "allenai/led-base-16384"

tokenizer = LEDTokenizer.from_pretrained(
    summarizer_name
)

model = LEDForConditionalGeneration.from_pretrained(
    summarizer_name
)

# =========================================================
# LOAD FAST NLI MODEL
# =========================================================

print("\nLoading NLI model...\n")

nli_pipeline = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
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
# SOURCE DOCUMENT
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
# GENERATE SUMMARY
# =========================================================

print("\nGenerating summary...\n")

inputs = tokenizer(
    source_document,
    return_tensors="pt",
    truncation=True,
    max_length=4096
)

# =========================================================
# GLOBAL ATTENTION MASK
# =========================================================

global_attention_mask = inputs["input_ids"].new_zeros(
    inputs["input_ids"].shape
)

global_attention_mask[:, 0] = 1

# =========================================================
# GENERATE SUMMARY
# =========================================================

summary_ids = model.generate(
    inputs["input_ids"],
    global_attention_mask=global_attention_mask,
    num_beams=2,
    max_length=80,
    min_length=20,
    early_stopping=True
)

generated_summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

# =========================================================
# PRINT SUMMARY
# =========================================================

print("\n===================================")
print("GENERATED SUMMARY")
print("===================================\n")

print(generated_summary)

# =========================================================
# SPLIT SUMMARY SENTENCES
# =========================================================

summary_sentences = nltk.sent_tokenize(
    generated_summary
)

# =========================================================
# PRECOMPUTE SOURCE EMBEDDING
# =========================================================

document_embedding = embed_model.encode(
    source_document,
    convert_to_tensor=True
)

# =========================================================
# TRUE NLI ANALYSIS
# =========================================================

sentence_scores = []

print("\n===================================")
print("TRUE NLI ANALYSIS")
print("===================================\n")

# =========================================================
# PROCESS SUMMARY SENTENCES
# =========================================================

for idx, summary_sent in enumerate(summary_sentences):

    print(f"\nSentence {idx+1}:")
    print(summary_sent)

    # =====================================================
    # FIXED NLI FORMAT
    # =====================================================

    result = nli_pipeline(
        f"{source_document} </s></s> {summary_sent}"
    )[0]

    label = result["label"]

    nli_score = result["score"]

    print(f"\nNLI Label:")
    print(label)

    print(f"\nNLI Score:")
    print(round(nli_score, 4))

    # =====================================================
    # SEMANTIC SIMILARITY
    # =====================================================

    summary_embedding = embed_model.encode(
        summary_sent,
        convert_to_tensor=True
    )

    semantic_score = util.cos_sim(
        document_embedding,
        summary_embedding
    ).item()

    print(f"\nSemantic Similarity:")
    print(round(semantic_score, 4))

    # =====================================================
    # FINAL SCORE
    # =====================================================

    if label == "ENTAILMENT":

        entailment_score = nli_score

    else:

        entailment_score = 0

    final_score = (
        0.7 * entailment_score
        +
        0.3 * semantic_score
    )

    final_score = max(0, min(final_score, 1))

    sentence_scores.append(final_score)

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
# FINAL SCORE
# =========================================================

true_score = np.mean(
    sentence_scores
)

hallucination_score = (
    1 - true_score
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

print(f"TRUE Score          : {true_score:.4f}")

print(f"Hallucination Score : {hallucination_score:.4f}")

print(f"Hallucination Rate  : {hallucination_rate:.2f}%")

# =========================================================
# FINAL DECISION
# =========================================================

print("\n===================================")
print("FINAL DECISION")
print("===================================\n")

if true_score >= 0.75:

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
    "fast_true_led_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("GENERATED SUMMARY\n")
    file.write("=================\n\n")

    file.write(generated_summary + "\n\n")

    file.write(f"TRUE Score: {true_score:.4f}\n")

    file.write(f"Hallucination Score: {hallucination_score:.4f}\n")

    file.write(f"Hallucination Rate: {hallucination_rate:.2f}%\n")

print("\nResults saved successfully.")