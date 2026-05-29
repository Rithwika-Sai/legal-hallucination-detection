from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
# -----------------------------------
# LOAD LED MODEL
# -----------------------------------

model_name = "allenai/led-base-16384"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# -----------------------------------
# READ LEGAL JUDGMENT
# -----------------------------------

with open("data/test.json", "r", encoding="utf-8") as f:
    text = f.read()

# LED can handle longer inputs
text = text[:8000]

# -----------------------------------
# TOKENIZE
# -----------------------------------

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=4096
)

# -----------------------------------
# GENERATE SUMMARY
# -----------------------------------

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=200,
    min_length=80,
    num_beams=4,
    early_stopping=True
)

generated_summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

# -----------------------------------
# PRINT
# -----------------------------------

print("\n===== LED SUMMARY =====\n")
print(generated_summary)

# -----------------------------------
# SAVE
# -----------------------------------

with open("outputs/led_output.txt", "w", encoding="utf-8") as f:
    f.write(generated_summary)

print("\nSummary saved successfully.")