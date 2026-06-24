This project addresses the critical vulnerability of "Hallucinate at the Last" — the empirically proven phenomenon where Large Language Models (LLMs) concentrate factual errors disproportionately in the concluding segments of generated long responses.
🏛️ Domain
Dataset: 100 complex Indian Supreme Court judgments
Task: Long document summarization (Background → Ratio Decidendi → Final Decision)
Challenge: Dense legal vocabulary, multi-page contexts, archaic phrasing, and strict factual constraints (dates, sections, citations)
Feature
Description
🎯 Positional Role Partitioning
Applies tiered mitigation strictness: First < Middle < Last
🔍 Retrieve-Then-Verify
Localized top-k evidence retrieval before NLI verification (inspired by CLEGHM)
🧬 Surgical Excision
Regex + NER-based removal of hallucinated legal atoms (dates, sections, entities)
🔁 NLI-Gated Retries
Evidence-constrained LLM rewrites with position-specific thresholds
📊 Coverage Healing
Reintegrates omitted high-fact-density source sentences
📈 Positional Trend Analysis
Tracks hallucination distribution across First/Middle/Last bins
