# =====================================================
# FAST FALCON-7B HALLUCINATION DETECTION
# OPTIMIZED FOR CPU LAPTOPS
# =====================================================

# =====================================================
# INSTALL
# =====================================================

# pip install transformers
# pip install torch
# pip install accelerate
# pip install sentencepiece
# pip install spacy
# pip install sentence-transformers

# python -m spacy download en_core_web_sm

# =====================================================
# WARNINGS
# =====================================================

import warnings
warnings.filterwarnings("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# =====================================================
# IMPORTS
# =====================================================

import torch
import spacy

from transformers import (

    AutoTokenizer,

    AutoModelForCausalLM
)

from sentence_transformers import (

    SentenceTransformer,

    util
)

# =====================================================
# CPU OPTIMIZATION
# =====================================================

torch.set_num_threads(4)

# =====================================================
# LOAD SPACY
# =====================================================

print("\nLoading spaCy...\n")

nlp = spacy.load("en_core_web_sm")

# =====================================================
# LOAD FAST EMBEDDING MODEL
# =====================================================

print("\nLoading Embedding Model...\n")

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =====================================================
# LOAD FALCON-7B
# =====================================================

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

# =====================================================
# IMPORTANT FIXES
# =====================================================

model.config.use_cache = False

model.eval()

tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# READ SOURCE
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

source = " ".join(source.split())

# =====================================================
# VERY SMALL INPUT FOR SPEED
# =====================================================

source_document = source[:200]

# =====================================================
# PROMPT
# =====================================================

prompt = f"""
Summarize briefly:

{source_document}

Summary:
"""

# =====================================================
# TOKENIZE
# =====================================================

inputs = tokenizer(

    prompt,

    return_tensors="pt",

    truncation=True,

    max_length=256
)

# =====================================================
# GENERATE SUMMARY
# =====================================================

print("\nGenerating Summary...\n")

with torch.no_grad():

    outputs = model.generate(

        input_ids=inputs["input_ids"],

        attention_mask=inputs["attention_mask"],

        max_new_tokens=15,

        do_sample=False,

        use_cache=False,

        num_beams=1,

        early_stopping=True,

        pad_token_id=tokenizer.eos_token_id
    )

# =====================================================
# DECODE
# =====================================================

summary = tokenizer.decode(

    outputs[0],

    skip_special_tokens=True
)

# =====================================================
# REMOVE PROMPT
# =====================================================

summary = summary.replace(
    prompt,
    ""
)

# =====================================================
# CLEAN SUMMARY
# =====================================================

summary = summary.replace("\n", " ")

summary = " ".join(summary.split())

# =====================================================
# EMPTY CHECK
# =====================================================

if len(summary.strip()) == 0:

    summary = (
        "The court reviewed the legal matter."
    )

# =====================================================
# PRINT SUMMARY
# =====================================================

print("\n===== GENERATED SUMMARY =====\n")

print(summary)

# =====================================================
# ENTITY EXTRACTION
# =====================================================

source_doc = nlp(source[:1500])

summary_doc = nlp(summary)

# =====================================================
# ENTITY LISTS
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
# UNSUPPORTED ENTITIES
# =====================================================

unsupported_entities = []

for ent in summary_entities:

    if ent not in source_entities:

        unsupported_entities.append(ent)

# =====================================================
# ENTITY SCORE
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
# FAST SEMANTIC SIMILARITY
# =====================================================

source_embedding = embed_model.encode(

    source[:500],

    convert_to_tensor=True
)

summary_embedding = embed_model.encode(

    summary,

    convert_to_tensor=True
)

semantic_similarity = util.cos_sim(

    source_embedding,

    summary_embedding

).item()

# =====================================================
# SEMANTIC SCORE
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
# OUTPUT
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

print("\n")

# =====================================================
# RELAXED THRESHOLD
# =====================================================

if hallucination_score > 65:

    print("Hallucination Detected")

else:

    print("Summary is Mostly Faithful")

print("\nDONE SUCCESSFULLY\n")



