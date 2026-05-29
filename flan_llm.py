import json
from transformers import pipeline
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
print("Loading BART model...")

pipe = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

print("Model loaded!")

# Load dataset
with open("data/test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract judgment
text = data[0]["judgment"]

# Skip metadata section
text = text[1500:4000]

print("Generating summary...")

# Generate summary
result = pipe(
    text,
    max_length=120,
    min_length=40,
    do_sample=False
)

summary = result[0]["summary_text"]

print("\n===== GENERATED SUMMARY =====\n")

print(summary)

# Save output
with open("outputs/bart_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)