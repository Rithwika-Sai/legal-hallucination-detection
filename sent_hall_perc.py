# =========================================================
# FAST SENTENCE-LEVEL HALLUCINATION POSITION ANALYSIS
# FOR:
# BART
# LED
# PEGASUS
# MISTRAL-7B
# FALCON-7B
# =========================================================

# =========================================================
# INSTALL
# =========================================================

# pip install transformers
# pip install torch
# pip install nltk
# pip install pandas
# pip install summac

# =========================================================
# IMPORTS
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from nltk.tokenize import sent_tokenize
import nltk

from transformers import pipeline

from summac.model_summac import SummaCZS

# =========================================================
# DOWNLOAD NLTK
# =========================================================

nltk.download("punkt")

# =========================================================
# LOAD TEST DOCUMENT
# =========================================================

with open(
    "data/test.txt",
    "r",
    encoding="utf-8"
) as f:

    source_document = f.read()

# =========================================================
# CLEAN DOCUMENT
# =========================================================

source_document = source_document.replace(
    "\n",
    " "
)

source_document = " ".join(
    source_document.split()
)

# =========================================================
# LIMIT SIZE FOR SPEED
# =========================================================

source_document = source_document[:1200]

# =========================================================
# LOAD MODELS
# =========================================================

print("\nLoading models...\n")

models = {

    "BART": pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    ),

    "LED": pipeline(
        "summarization",
        model="allenai/led-base-16384"
    ),

    "PEGASUS": pipeline(
        "summarization",
        model="google/pegasus-xsum"
    )
}

# =========================================================
# OPTIONAL LARGE MODELS
# =========================================================

try:

    models["MISTRAL-7B"] = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        device_map="auto"
    )

except:
    print("Skipping MISTRAL-7B")

try:

    models["FALCON-7B"] = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        device_map="auto"
    )

except:
    print("Skipping FALCON-7B")

# =========================================================
# LOAD SUMMAC
# =========================================================

print("\nLoading SummaC...\n")

summac_model = SummaCZS(
    granularity="sentence",
    model_name="vitc"
)

# =========================================================
# THRESHOLD
# =========================================================

threshold = 0.5

# =========================================================
# RESULTS
# =========================================================

results = []

# =========================================================
# MAIN LOOP
# =========================================================

for model_name, model in models.items():

    print(f"\nRunning {model_name}...\n")

    try:

        # =================================================
        # GENERATE SUMMARY
        # =================================================

        if model_name in ["BART", "LED", "PEGASUS"]:

            summary = model(

                source_document,

                max_length=80,

                min_length=25,

                truncation=True,

                do_sample=False

            )[0]["summary_text"]

        else:

            prompt = f"""
            Summarize this legal judgment briefly:

            {source_document}

            Summary:
            """

            output = model(

                prompt,

                max_new_tokens=80,

                do_sample=False,

                temperature=0.2

            )

            summary = output[0]["generated_text"]

            summary = summary.replace(
                prompt,
                ""
            )

        # =================================================
        # TOKENIZE SENTENCES
        # =================================================

        sentences = sent_tokenize(summary)

        # Skip very short summaries
        if len(sentences) < 3:
            continue

        first_sentence = [sentences[0]]

        middle_sentences = sentences[1:-1]

        last_sentence = [sentences[-1]]

        # =================================================
        # COUNTERS
        # =================================================

        first_hall = 0
        middle_hall = 0
        last_hall = 0

        # =================================================
        # FIRST SENTENCE
        # =================================================

        for sent in first_sentence:

            score = summac_model.score(

                [source_document],

                [sent]

            )["scores"][0]

            if score < threshold:

                first_hall += 1

        # =================================================
        # MIDDLE SENTENCES
        # =================================================

        for sent in middle_sentences:

            score = summac_model.score(

                [source_document],

                [sent]

            )["scores"][0]

            if score < threshold:

                middle_hall += 1

        # =================================================
        # LAST SENTENCE
        # =================================================

        for sent in last_sentence:

            score = summac_model.score(

                [source_document],

                [sent]

            )["scores"][0]

            if score < threshold:

                last_hall += 1

        # =================================================
        # TOTAL HALLUCINATIONS
        # =================================================

        total_hall = (

            first_hall +

            middle_hall +

            last_hall
        )

        # Avoid divide-by-zero
        if total_hall == 0:
            total_hall = 1

        # =================================================
        # PERCENTAGES
        # =================================================

        first_percent = (
            first_hall / total_hall
        ) * 100

        middle_percent = (
            middle_hall / total_hall
        ) * 100

        last_percent = (
            last_hall / total_hall
        ) * 100

        # =================================================
        # STORE RESULTS
        # =================================================

        results.append({

            "Model": model_name,

            "First sentence":
            round(first_percent, 2),

            "Middle sentences":
            round(middle_percent, 2),

            "Last sentence":
            round(last_percent, 2)
        })

        # =================================================
        # PRINT SUMMARY
        # =================================================

        print("\nSUMMARY:\n")

        print(summary)

        print("\n" + "="*60)

    except Exception as e:

        print(f"\nError in {model_name}: {e}")

# =========================================================
# FINAL TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("="*70)
print("TABLE 7 — SENTENCE-LEVEL HALLUCINATION POSITION")
print("="*70)

print(results_df)












# =========================================================
# SAVE CSV
# =========================================================

results_df.to_csv(

    "table7_hallucination_positions.csv",

    index=False
)

print("\nSaved:")
print("table7_hallucination_positions.csv")

print("\nDONE SUCCESSFULLY\n")