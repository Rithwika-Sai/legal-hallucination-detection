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
# -----------------------------------
# LOAD PEGASUS MODEL
# -----------------------------------

summarizer = pipeline(
    "summarization",
    model="google/pegasus-xsum"
)

# -----------------------------------
# READ JUDGMENT
# -----------------------------------

with open("data/test.json", "r", encoding="utf-8") as f:
    text = f.read()

# limit size
text = text[:1200]

# -----------------------------------
# GENERATE SUMMARY
# -----------------------------------

summary = summarizer(
    text,
    max_length=200,
    min_length=80,
    do_sample=False
)

generated_summary = summary[0]['summary_text']

# -----------------------------------
# PRINT
# -----------------------------------

print("\n===== PEGASUS SUMMARY =====\n")
print(generated_summary)

# -----------------------------------
# SAVE
# -----------------------------------

with open("outputs/pegasus_output.txt", "w", encoding="utf-8") as f:
    f.write(generated_summary)

print("\nSummary saved successfully.")