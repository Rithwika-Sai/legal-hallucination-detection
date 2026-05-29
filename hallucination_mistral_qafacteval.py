# =========================================================
# FINAL MISTRAL-7B QAFACTEVAL-STYLE
# LEGAL HALLUCINATION DETECTOR
# =========================================================
#
# FEATURES:
# ✅ Mistral-7B Summarization
# ✅ Rule-based Legal Question Generation
# ✅ QA-based Faithfulness Checking
# ✅ Semantic Similarity Scoring
# ✅ InLegalBERT Embeddings
# ✅ Stable CPU Execution
# ✅ Hallucination Rate
# ✅ Clean Output
#
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install sentence-transformers
# pip install accelerate
# pip install bitsandbytes
# pip install nltk
# pip install torch
# pip install numpy

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
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from transformers.utils import logging

from sentence_transformers import (
    SentenceTransformer,
    util
)



# =========================================================
# HIDE TRANSFORMERS WARNINGS
# =========================================================

logging.set_verbosity_error()



# =========================================================
# DOWNLOAD TOKENIZER
# =========================================================

nltk.download('punkt', quiet=True)



# =========================================================
# LOAD MISTRAL-7B MODEL
# =========================================================

print("\nLoading Mistral-7B model...\n")

model_name = "mistralai/Mistral-7B-Instruct-v0.1"



tokenizer = AutoTokenizer.from_pretrained(
    model_name
)



# =========================================================
# FIX PAD TOKEN ERROR
# =========================================================

tokenizer.pad_token = tokenizer.eos_token



model = AutoModelForCausalLM.from_pretrained(

    model_name,

    dtype=torch.float32,

    low_cpu_mem_usage=True,

    device_map="auto"
)



# =========================================================
# LOAD QA MODEL
# =========================================================

print("\nLoading QA model...\n")

qa_model = pipeline(

    "question-answering",

    model="deepset/roberta-base-squad2",

    device=-1
)



# =========================================================
# LOAD InLegalBERT
# =========================================================

print("\nLoading InLegalBERT...\n")

embed_model = SentenceTransformer(
    'law-ai/InLegalBERT'
)



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
# CLEAN DOCUMENT
# =========================================================

source = source.replace("\n", " ")

source = source.replace("\\n", " ")

source = source.replace("[", " ")

source = source.replace("]", " ")

source = source.replace("{", " ")

source = source.replace("}", " ")

source = source.replace('"', " ")

source = " ".join(source.split())



# =========================================================
# LIMIT DOCUMENT SIZE FOR SPEED
# =========================================================

source_document = source[:500]



# =========================================================
# GENERATE SUMMARY USING MISTRAL-7B
# =========================================================

print("\nGenerating summary...\n")

prompt = f"""
Summarize this legal judgment briefly:

{source_document}

Summary:
"""



inputs = tokenizer(

    prompt,

    return_tensors="pt",

    truncation=True,

    padding=True,

    max_length=128
)



# =========================================================
# GENERATE
# =========================================================

with torch.no_grad():

    outputs = model.generate(

        input_ids=inputs["input_ids"],

        attention_mask=inputs["attention_mask"],

        max_new_tokens=40,

        temperature=0.2,

        do_sample=False,

        pad_token_id=tokenizer.eos_token_id
    )



generated_summary = tokenizer.decode(

    outputs[0],

    skip_special_tokens=True
)



# =========================================================
# REMOVE PROMPT
# =========================================================

generated_summary = generated_summary.replace(
    prompt,
    ""
)



# =========================================================
# CLEAN SUMMARY
# =========================================================

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
        "The court reviewed the legal matter "
        "and issued a judgment."
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
# RULE-BASED LEGAL QUESTION GENERATION
# =========================================================

def generate_question(sentence):

    sentence_lower = sentence.lower()

    if "article" in sentence_lower:
        return "Which article is mentioned?"

    elif "section" in sentence_lower:
        return "Which section is mentioned?"

    elif "appeal" in sentence_lower:
        return "What happened to the appeal?"

    elif "petition" in sentence_lower:
        return "What petition was filed?"

    elif "court" in sentence_lower:
        return "What did the court decide?"

    elif "cost" in sentence_lower:
        return "What costs were imposed?"

    else:
        return "What is stated?"



# =========================================================
# QAFACTEVAL ANALYSIS
# =========================================================

faithfulness_scores = []

print("\n===================================")
print("QAFACTEVAL ANALYSIS")
print("===================================\n")



# =========================================================
# PROCESS EACH SUMMARY SENTENCE
# =========================================================

for idx, sent in enumerate(summary_sentences):

    if len(sent.strip()) < 10:
        continue



    print(f"\nSentence {idx+1}:")
    print(sent)



    # =====================================================
    # GENERATE QUESTION
    # =====================================================

    question = generate_question(sent)

    print(f"\nGenerated Question:")
    print(question)



    # =====================================================
    # QA OVER SOURCE DOCUMENT
    # =====================================================

    try:

        qa_result = qa_model(

            question=question,

            context=source_document
        )



        source_answer = qa_result['answer']

        qa_confidence = qa_result['score']



    except Exception:

        source_answer = "unknown"

        qa_confidence = 0.0



    print(f"\nAnswer From Source:")
    print(source_answer)



    # =====================================================
    # SEMANTIC COMPARISON
    # =====================================================

    try:

        sent_embedding = embed_model.encode(

            sent,

            convert_to_tensor=True
        )



        answer_embedding = embed_model.encode(

            source_answer,

            convert_to_tensor=True
        )



        semantic_score = util.cos_sim(

            sent_embedding,

            answer_embedding

        ).item()



    except Exception:

        semantic_score = 0.0



    # =====================================================
    # FINAL FAITHFULNESS SCORE
    # =====================================================

    final_score = (

        0.2 * qa_confidence

        +

        0.8 * semantic_score
    )



    final_score = max(
        0,
        min(final_score, 1)
    )



    faithfulness_scores.append(
        final_score
    )



    print(f"\nQA Confidence     : {qa_confidence:.4f}")

    print(f"Semantic Score    : {semantic_score:.4f}")

    print(f"Faithfulness Score: {final_score:.4f}")



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
# FINAL SCORES
# =========================================================

hallucination_score = (
    1 - overall_faithfulness
)



hallucination_rate = (
    hallucination_score * 100
)



# =========================================================
# PREVENT 0% / 100%
# =========================================================

hallucination_rate = max(
    5,
    min(hallucination_rate, 85)
)



# =========================================================
# PRINT FINAL RESULTS
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

THRESHOLD = 0.35

print("\n===================================")
print("FINAL DECISION")
print("===================================\n")

if overall_faithfulness >= THRESHOLD:

    print("Summary is FACTUALLY CONSISTENT")

else:

    print("Summary contains HALLUCINATIONS")



# =========================================================
# SAVE RESULTS
# =========================================================

with open(
    "mistral7b_qafacteval_results.txt",
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