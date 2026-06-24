This project addresses the critical vulnerability of "Hallucinate at the Last" — the empirically proven phenomenon where Large Language Models (LLMs) concentrate factual errors disproportionately in the concluding segments of generated long responses.


🏛️ Domain


Dataset: 100 complex Indian Supreme Court judgments


Task: Long document summarization (Background → Ratio Decidendi → Final Decision)


Challenge: Dense legal vocabulary, multi-page contexts, archaic phrasing, and strict factual constraints (dates, sections, citations)






┌──────────────────────────────────────────────────────────────┐
│              INPUT: 100 Indian Supreme Court Judgments       │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│   PHASE 1: Mistral-7B-Instruct-v0.3 (Zero-Shot Generation) │
│   • Structured prompt (Background/Ratio/Decision)            │
│   • Greedy decoding, repetition_penalty=1.15                 │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│   PHASE 2: Positional Partitioning & Evidence Retrieval      │
│   • First  → Top-k=2 evidence passages                       │
│   • Middle → Top-k=3 evidence passages                       │
│   • Last   → Top-k=4 evidence passages                       │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│   PHASE 3: Position-Aware NLI Mitigation                     │
│   • DeBERTa-v3 NLI entailment checking                       │
│   • Thresholds: τ_first=0.0, τ_mid=0.40, τ_last=0.50        │
│   • Retries: R_first=0, R_mid=2, R_last=3                    │
│   • Strong Mode Fallback for unresolvable claims             │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│   PHASE 4: Coverage Healing (Omission Resolution)            │
│   • Identifies high-density omitted facts (COV_THRESH=0.60)  │
│   • Appends top-4 critical omissions                         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│   PHASE 5: Final Metric Aggregation & Trend Analysis         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│   OUTPUT: Verified Legal Summaries + Evaluation Report       │
└──────────────────────────────────────────────────────────────┘
