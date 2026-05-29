from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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
# -----------------------------
# LOAD MODEL
# -----------------------------

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

# -----------------------------
# READ LEGAL JUDGMENT
# -----------------------------

with open("data/test.json", "r", encoding="utf-8") as f:
    text = f.read()

# limit long text
text = text[:2500]

# -----------------------------
# PROMPT
# -----------------------------

prompt = f"""
Summarize the following legal judgment.

Focus on:
- parties
- legal issue
- court reasoning
- final judgment

Judgment:
{text}

Summary:
"""

# -----------------------------
# TOKENIZE INPUT
# -----------------------------

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=2048
)

# -----------------------------
# GENERATE SUMMARY
# -----------------------------

outputs = model.generate(
    **inputs,
    max_new_tokens=400,
    temperature=0.3,
    do_sample=True,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

# -----------------------------
# DECODE OUTPUT
# -----------------------------

generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

# Remove prompt from output
generated_summary = generated_text[len(prompt):].strip()

# -----------------------------
# PRINT SUMMARY
# -----------------------------

print("\n===== PHI-2 SUMMARY =====\n")
print(generated_summary)

# -----------------------------
# SAVE SUMMARY
# -----------------------------

with open("outputs/phi2_output.txt", "w", encoding="utf-8") as f:
    f.write(generated_summary)

print("\nSummary saved successfully.")