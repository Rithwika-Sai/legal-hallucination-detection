import json
import os
import math
import re
import torch
from rank_bm25 import BM25Okapi

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
JSON_PATH  = "data/test.json"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Switch between "falcon" and "mistral" here
MODEL_ENGINE = "mistral"   

MODEL_IDS = {
    "falcon" : "tiiuae/falcon-7b-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}
MODEL_ID = MODEL_IDS[MODEL_ENGINE]

# ─── POSITIONAL WEIGHT FUNCTION ──────────────────────────────────────────────
def positional_weight(sentence_index, total_sentences):
    """
    Gaussian bell curve centred at the middle of a passage.
    - Centre sentence  → 1.0   (highest preference)
    - First/last sent  → MIN_W (deprioritised, not silenced)

    Tune:
      MIN_W  ↑ → flatter curve (edges matter more)
      SIGMA  ↓ → sharper peak  (only very centre matters)
    """
    MIN_W = 0.3
    SIGMA = 0.25

    if total_sentences == 1:
        return 1.0

    normalised = sentence_index / (total_sentences - 1)
    gaussian   = math.exp(-((normalised - 0.5) ** 2) / (2 * SIGMA ** 2))
    return round(MIN_W + (1.0 - MIN_W) * gaussian, 4)

# ─── POSITIONAL ROLE RETRIEVER ────────────────────────────────────────────────
class PositionalRoleRetriever:
    """
    BM25 retriever with per-role indexes and positional re-ranking.
    final_score = bm25_score × positional_weight(sentence_index)
    """

    ROLES = ["facts", "arguments", "ratio_decidendi", "relief"]

    def __init__(self, case_data: dict):
        self.sentences  = {}
        self.bm25_index = {}
        self._build_indexes(case_data)

    def _split_sentences(self, text: str):
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw if s.strip()]

    def _build_indexes(self, case_data: dict):
        for role in self.ROLES:
            sents = self._split_sentences(case_data.get(role, ""))
            self.sentences[role]  = sents
            tokenised = [s.lower().split() for s in sents]
            self.bm25_index[role] = BM25Okapi(tokenised) if tokenised else None

    def retrieve(self, query: str, role: str, top_k: int = 3, verbose: bool = True):
        sents = self.sentences.get(role, [])
        index = self.bm25_index.get(role)
        if not sents or index is None:
            return []

        n           = len(sents)
        bm25_scores = index.get_scores(query.lower().split())

        scored = []
        for i, (sent, raw) in enumerate(zip(sents, bm25_scores)):
            pos_w = positional_weight(i, n)
            scored.append({
                "sentence"   : sent,
                "position"   : i,
                "total"      : n,
                "bm25_raw"   : round(float(raw), 4),
                "pos_weight" : pos_w,
                "final_score": round(float(raw) * pos_w, 4),
            })

        ranked = sorted(scored, key=lambda x: x["final_score"], reverse=True)

        if verbose:
            print(f"\n  {'Pos':>3}  {'BM25':>7}  {'PosW':>6}  {'Final':>7}  Sentence")
            print(f"  {'---':>3}  {'-------':>7}  {'------':>6}  {'-------':>7}  --------")
            for r in ranked:
                snippet = r["sentence"][:55] + ("…" if len(r["sentence"]) > 55 else "")
                print(f"  {r['position']:>3}  {r['bm25_raw']:>7.4f}  "
                      f"{r['pos_weight']:>6.3f}  {r['final_score']:>7.4f}  {snippet}")

        return ranked[:top_k]

# ─── PROMPT BUILDER ───────────────────────────────────────────────────────────
def build_prompt(engine: str, role: str, context: str, query: str) -> str:
    """
    Constructs model-specific prompt from retrieved sentences.
    Falcon uses System/User/Assistant format.
    Mistral uses <s>[INST] ... [/INST] format.
    """
    system = (
        "You are a zero-trust legal reader. "
        "Answer ONLY from the provided context sentences. "
        "If the answer is not present, say 'Information not found within this role.'"
    )
    user = (
        f"You are evaluating the '{role.upper()}' segment of a legal case.\n"
        f"Retrieved context sentences:\n{context}\n\n"
        f"Question: {query}"
    )

    if engine == "falcon":
        return f"System: {system}\nUser: {user}\nAssistant:"
    elif engine == "mistral":
        return f"<s>[INST] {system}\n{user} [/INST]"
    return f"{user}"

# ─── MOCK GENERATOR (swap for real model.generate on GPU) ────────────────────
def mock_generate(engine: str, role: str, context_sentences: list, query: str) -> str:
    """
    Simulates model output from retrieved positional sentences.
    In production: replace this entire function with real tokenizer + model.generate().

    Real implementation would be:
        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
        out_ids = model.generate(**inputs, max_new_tokens=150,
                                 temperature=0.1, use_cache=True,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:],
                                skip_special_tokens=True).strip()
    """
    # Build the prompt so you can inspect the exact string sent to the model
    context_str = " ".join([r["sentence"] for r in context_sentences])
    prompt      = build_prompt(engine, role, context_str, query)

    print(f"\n  [PROMPT SENT TO {engine.upper()}]")
    print(f"  {'-'*58}")
    for line in prompt.split("\n"):
        print(f"  {line}")
    print(f"  {'-'*58}")

    # Simulated response (replace with real model output)
    responses = {
        ("falcon", "arguments"): (
            "The defense counsel argued that the eyewitness testimonies were "
            "highly contradictory and unreliable. Furthermore, counsel submitted "
            "that no forensic evidence links the accused to the weapon."
        ),
        ("mistral", "arguments"): (
            "Based on the retrieved context, defense counsel contended that the "
            "eyewitness accounts were contradictory and unreliable, and additionally "
            "submitted that no forensic evidence connects the accused to the weapon."
        ),
        ("falcon", "ratio_decidendi"): (
            "The court held that corroborative circumstantial evidence takes "
            "precedence, and eyewitness reliability is secondary when such "
            "evidence is solid. Hostile witnesses were evaluated within the "
            "overall evidence matrix."
        ),
        ("mistral", "ratio_decidendi"): (
            "According to the retrieved context, the court held that eyewitness "
            "reliability is secondary when corroborative circumstantial evidence "
            "is solid. The hostile witnesses were considered alongside the full "
            "evidence matrix."
        ),
    }
    return responses.get((engine, role), "Information not found within this role.")

# ─── DATA LOADER ─────────────────────────────────────────────────────────────
def load_case_data(json_path: str) -> dict:
    if not os.path.exists(json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        sample = [{
            "facts": (
                "The appellant was convicted under Section 302 IPC by the sessions court. "
                "He was accused of murdering his neighbour during a land dispute. "
                "The incident occurred on the night of 14th March. "
                "Police arrived at the scene within two hours of the incident."
            ),
            "arguments": (
                "Defense counsel opens by questioning the investigation procedure. "
                "Defense counsel argues that the eyewitness testimonies are highly contradictory and unreliable. "
                "Main prosecution witnesses turned hostile during cross-examination. "
                "Counsel further submits that no forensic evidence links the accused to the weapon. "
                "The closing submission requests acquittal on grounds of reasonable doubt."
            ),
            "ratio_decidendi": (
                "The court noted that minor contradictions do not erode a prosecution case. "
                "It held that if corroborative circumstantial evidence is solid, eyewitness reliability is secondary. "
                "Hostile witnesses were considered in light of the overall evidence matrix. "
                "The court concluded that the conviction is sustainable on available material."
            ),
            "relief": (
                "The criminal appeal is hereby dismissed. "
                "The conviction and sentence passed by the sessions court are upheld."
            ),
        }]
        with open(json_path, "w") as f:
            json.dump(sample, f, indent=4)
        print(f"[DATA] Created sample data at {json_path}")

    with open(json_path) as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) else data

# ─── WEIGHT CURVE DISPLAY ─────────────────────────────────────────────────────
def show_weight_curve(n: int = 7):
    print(f"\n  Positional weight curve  ({n} sentences):")
    print(f"  {'Pos':>4}  {'Label':>10}  {'Weight':>8}  Bar")
    print(f"  {'----':>4}  {'----------':>10}  {'--------':>8}  ---")
    for i in range(n):
        w     = positional_weight(i, n)
        label = ("← first" if i == 0 else
                 "← last"  if i == n - 1 else
                 "← CENTRE" if i == n // 2 else "")
        print(f"  {i:>4}  {label:>10}  {w:>8.4f}  {'█' * int(w * 20)}")

# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print(f"[CONFIG] Model engine : {MODEL_ENGINE.upper()} ({MODEL_ID})")
    print(f"[CONFIG] Device       : {DEVICE}")

    # ── Load model weights (comment out for mock mode) ────────────────────────
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # print(f"[LOAD] Loading tokenizer and model...")
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_ID,
    #     dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    #     device_map="auto" if DEVICE == "cuda" else None,
    #     low_cpu_mem_usage=True,
    #     attn_implementation="sdpa" if DEVICE == "cuda" else "eager",
    #     trust_remote_code=(MODEL_ENGINE == "falcon"),
    # )
    # print(f"[LOAD] Model ready.")

    # ── Load data and build retriever ─────────────────────────────────────────
    print(f"\n[STEP 1] Loading case data from {JSON_PATH}...")
    case_data = load_case_data(JSON_PATH)

    print(f"[STEP 2] Building positional BM25 indexes per role...")
    retriever = PositionalRoleRetriever(case_data)

    show_weight_curve(n=7)

    # ── Query 1 : arguments ───────────────────────────────────────────────────
    query1 = "What did the defense counsel argue regarding the witnesses?"
    role1  = "arguments"

    print(f"\n{'═'*64}")
    print(f"[STEP 3] Retrieving from role: {role1.upper()}")
    print(f"  Query : {query1}")
    print(f"{'═'*64}")

    top1    = retriever.retrieve(query1, role=role1, top_k=2)
    answer1 = mock_generate(MODEL_ENGINE, role1, top1, query1)

    print(f"\n{'='*64}")
    print(f"ROLE-PARTITIONED PIPELINE OUTPUT  [{MODEL_ENGINE.upper()}]")
    print(f"{'='*64}")
    print(f"Target Role   : {role1.upper()}")
    print(f"Retrieved Sents: {len(top1)}  (positionally re-ranked)")
    print(f"Model Response :\n{answer1}")
    print(f"{'═'*64}")

    # ── Query 2 : ratio_decidendi ─────────────────────────────────────────────
    query2 = "What did the court hold about eyewitness reliability?"
    role2  = "ratio_decidendi"

    print(f"\n[STEP 4] Retrieving from role: {role2.upper()}")
    print(f"  Query : {query2}")
    print(f"{'═'*64}")

    top2    = retriever.retrieve(query2, role=role2, top_k=2)
    answer2 = mock_generate(MODEL_ENGINE, role2, top2, query2)

    print(f"\n{'='*64}")
    print(f"ROLE-PARTITIONED PIPELINE OUTPUT  [{MODEL_ENGINE.upper()}]")
    print(f"{'='*64}")
    print(f"Target Role   : {role2.upper()}")
    print(f"Retrieved Sents: {len(top2)}  (positionally re-ranked)")
    print(f"Model Response :\n{answer2}")
    print(f"{'═'*64}")

    print(f"\n[NOTE] To switch models, set MODEL_ENGINE = 'falcon' or 'mistral' at the top.")
    print(f"[NOTE] To use real inference, uncomment the model loading block above")
    print(f"       and replace mock_generate() calls with real tokenizer + model.generate().")