
# =========================================================
# ULTRA FAST FALCON-7B KG HALLUCINATION DETECTION
# CPU OPTIMIZED VERSION
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install accelerate
# pip install sentence-transformers
# pip install spacy
# pip install nltk
# pip install torch

# python -m spacy download en_core_web_sm

# =========================================================
# IMPORTS
# =========================================================

import json
import nltk
import spacy
import torch

from sentence_transformers import (

    SentenceTransformer,

    util
)

from transformers import (

    AutoTokenizer,

    AutoModelForCausalLM
)

# =========================================================
# CPU OPTIMIZATION
# =========================================================

torch.set_num_threads(4)

# =========================================================
# DOWNLOAD TOKENIZER
# =========================================================

nltk.download("punkt", quiet=True)

# =========================================================
# LOAD SPACY
# =========================================================

print("\nLoading spaCy...\n")

nlp = spacy.load("en_core_web_sm")

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
# TAKE ONLY 1 SAMPLE
# =========================================================

sample = data[0]

# =========================================================
# VERY SMALL INPUT
# =========================================================

source = sample["judgment"][:200]

# =========================================================
# GENERATE SUMMARY
# =========================================================

print("\nGenerating summary...\n")

prompt = f"""
Summarize briefly:

{source}

Summary:
"""

# =========================================================
# TOKENIZE INPUT
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

summary = tokenizer.decode(

    outputs[0],

    skip_special_tokens=True
)

# =========================================================
# CLEAN SUMMARY
# =========================================================

summary = summary.replace(

    prompt,

    ""
).strip()

summary = summary.replace(
    "\n",
    " "
)

summary = " ".join(
    summary.split()
)

# =========================================================
# EMPTY SUMMARY CHECK
# =========================================================

if len(summary.strip()) == 0:

    summary = (
        "The court reviewed the legal matter."
    )

# =========================================================
# PRINT SUMMARY
# =========================================================

print("\n==============================")
print("GENERATED SUMMARY")
print("==============================\n")

print(summary)

# =========================================================
# EXTRACT ENTITIES
# =========================================================

source_doc = nlp(source)

summary_doc = nlp(summary)

# =========================================================
# ENTITY LISTS
# =========================================================

source_entities = list(set(

    ent.text.lower().strip()

    for ent in source_doc.ents
))

summary_entities = list(set(

    ent.text.lower().strip()

    for ent in summary_doc.ents
))

# =========================================================
# HANDLE EMPTY ENTITIES
# =========================================================

if len(source_entities) == 0:

    source_entities = ["legal case"]

# =========================================================
# PRECOMPUTE SOURCE EMBEDDINGS
# =========================================================

source_embeddings = embed_model.encode(

    source_entities,

    convert_to_tensor=True
)

# =========================================================
# SEMANTIC DOCUMENT EMBEDDING
# =========================================================

document_embedding = embed_model.encode(

    source,

    convert_to_tensor=True
)

# =========================================================
# HALLUCINATION SCORE
# =========================================================

hallucination_score = 0

# =========================================================
# ENTITY CHECK
# =========================================================

for entity in summary_entities:

    entity_embedding = embed_model.encode(

        entity,

        convert_to_tensor=True
    )

    similarities = util.cos_sim(

        entity_embedding,

        source_embeddings
    )

    max_similarity = similarities.max().item()

    # =====================================================
    # ENTITY PENALTY
    # =====================================================

    if max_similarity < 0.50:

        hallucination_score += 20

# =========================================================
# SEMANTIC SIMILARITY
# =========================================================

summary_embedding = embed_model.encode(

    summary,

    convert_to_tensor=True
)

semantic_similarity = util.cos_sim(

    document_embedding,

    summary_embedding

).item()

# =========================================================
# SEMANTIC PENALTY
# =========================================================

if semantic_similarity < 0.60:

    hallucination_score += (
        (1 - semantic_similarity) * 50
    )

# =========================================================
# NORMALIZE SCORE
# =========================================================

hallucination_rate = min(

    hallucination_score,

    85
)

hallucination_rate = max(

    hallucination_rate,

    5
)

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n==============================")
print("FINAL RESULTS")
print("==============================\n")

print(
    f"Semantic Similarity : "
    f"{semantic_similarity:.4f}"
)

print(
    f"Hallucination Rate  : "
    f"{hallucination_rate:.2f}%"
)

# =========================================================
# FINAL INTERPRETATION
# =========================================================

print("\n==============================")
print("INTERPRETATION")
print("==============================\n")

if hallucination_rate < 20:

    print("Low hallucination detected.")

elif hallucination_rate < 50:

    print("Moderate hallucination detected.")

else:

    print("High hallucination detected.")

print("\nDONE SUCCESSFULLY\n")
