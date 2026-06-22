# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — PRECOMPUTE  (run ONCE in Colab, GPU runtime, no time limit)   ║
# ║                                                                          ║
# ║  This script does ALL the expensive work:                                ║
# ║    1. Embeds the JD (4 query facets)                                     ║
# ║    2. Embeds all ~20k candidates                                         ║
# ║    3. Runs semantic search -> top 300 candidates                         ║
# ║    4. Calls GROQ API for a one-line reason per top-300 candidate         ║
# ║       (each prompt is candidate-specific — see DIVERSITY FIX below)      ║
# ║                                                                          ║
# ║  FIX 1 — SWITCHED TO GROQ                                                ║
# ║  ────────────────────────────────────────────────────                    ║
# ║  Groq's free tier is far more generous than Gemini's (30 RPM on         ║
# ║  llama-3.3-70b-versatile vs Gemini's ~10-15 RPM), so the same pacing +   ║
# ║  backoff pattern now runs faster with fewer retries needed.              ║
# ║                                                                          ║
# ║  FIX 2 — REASONING DIVERSITY                                             ║
# ║  ────────────────────────────────────────────────────                    ║
# ║  Each prompt now injects the candidate's SPECIFIC company, system        ║
# ║  type, scale numbers and skill-assessment results, and explicitly        ║
# ║  instructs the model to anchor on those specific details rather than     ║
# ║  general phrasing — so two different candidates can't get the same      ║
# ║  generic "HR Manager with X yrs" templated sentence.                     ║
# ║                                                                          ║
# ║  This is run SEPARATELY from your final 5-min CPU-only submission run.  ║
# ║  Run this whenever you want, on GPU, and just save the outputs.          ║
# ║                                                                          ║
# ║  OUTPUTS (saved to OUTPUT_DIR):                                          ║
# ║    jd_embedding.npy              — JD vectors (4 facets × 384 dims)      ║
# ║    candidate_embeddings.npz      — all candidate vectors + IDs           ║
# ║    top300_with_reasons.json      — top 300 candidates, with:             ║
# ║                                     - semantic_score (precomputed)       ║
# ║                                     - llm_reason (precomputed, unique)   ║
# ║                                     - full original candidate data       ║
# ║                                                                          ║
# ║  STEP 0: Runtime → Change runtime type → T4 GPU                          ║
# ║  STEP 1: Mount Drive:                                                    ║
# ║            from google.colab import drive                                ║
# ║            drive.mount('/content/drive')                                 ║
# ║  STEP 2: Install:                                                        ║
# ║            !pip install sentence-transformers faiss-gpu groq --quiet     ║
# ║            # if faiss-gpu fails: !pip install faiss-cpu groq --quiet     ║
# ║  STEP 3: Set GROQ_API_KEY in Colab Secrets (🔑 icon) or below           ║
# ║  STEP 4: Update INPUT_PATH / OUTPUT_DIR, then Run All                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os, json, time, re
import numpy as np
from pathlib import Path
from datetime import datetime, date

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("Run: !pip install sentence-transformers --quiet")
try:
    import faiss
except ImportError:
    raise SystemExit("Run: !pip install faiss-gpu --quiet  (or faiss-cpu)")
try:
    from groq import Groq
except ImportError:
    raise SystemExit("Run: !pip install groq --quiet")
import torch

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

INPUT_PATH  = "againhoneyfiltered_candidates (1).json"   # your ~20k filtered file
OUTPUT_DIR  = "/content/drive/MyDrive/redrob/outputs"

GROQ_API_KEY = ""   # paste here, or leave blank to use Colab Secrets
GROQ_MODEL   = "llama-3.3-70b-versatile"   # strong free-tier model on Groq

MODEL_NAME     = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM  = 384
BATCH_SIZE_GPU = 512
BATCH_SIZE_CPU = 128

TOP_SEMANTIC = 300   # how many candidates to carry forward to scoring stage

# ── RATE LIMITING ─────────────────────────────────────────────────────────────
# Groq free tier on llama-3.3-70b-versatile: ~30 RPM. 2.2s/call = ~27 RPM,
# safely under the cap with margin. Raise if you're on a paid tier.
MIN_SECONDS_BETWEEN_CALLS = 2.2
MAX_RETRIES               = 5
INITIAL_BACKOFF_SECONDS   = 8

CHECKPOINT_EVERY = 20   # save progress every N candidates, supports resume

# ── 4 JD query facets — OR logic (candidate excelling at ANY facet is good) ──
JD_QUERIES = [
    (
        "Senior engineer who built and deployed production ranking retrieval "
        "or recommendation systems to real users at scale. Handles embedding drift "
        "index refresh retrieval quality regression in live production environments. "
        "Shipped end-to-end search or recommendation system. Owns system from design "
        "to deployment. Production experience with vector databases FAISS Elasticsearch "
        "Qdrant Pinecone Weaviate Milvus. Hybrid BM25 dense retrieval in production."
    ),
    (
        "Engineer who designed evaluation frameworks for ranking and retrieval systems. "
        "Built offline to online evaluation pipelines. NDCG MRR MAP precision recall "
        "A/B testing feedback loops. Knows difference between offline benchmark and "
        "online recruiter engagement metrics. Improved ranking system based on real "
        "user feedback data. Has defended retrieval architecture decisions with data."
    ),
    (
        "Applied machine learning engineer at product companies not consulting firms. "
        "5 to 8 years experience. Shipped ML systems to real users not just prototypes. "
        "Understands when to use BM25 versus dense retrieval. Prefers working system "
        "over perfect model. Pragmatic about LLM integration. Learning to rank XGBoost "
        "LightGBM. Sentence transformers BGE E5 in production pipelines."
    ),
    (
        "Engineer integrating LLMs into production retrieval and ranking pipelines. "
        "Knows when to fine-tune versus prompt engineer. Used sentence transformers "
        "OpenAI embeddings BGE E5 in production. Experience fine-tuning with LoRA "
        "QLoRA PEFT. Hybrid search combining dense and sparse retrieval. "
        "Actually understands retrieval quality not just calling OpenAI API."
    ),
]

# ═══════════════════════════════════════════════════════════════
#  TECHNICAL ROLE GATE  ← THE FIX FOR "HR MANAGER RANKED #1"
#  ──────────────────────────────────────────────────────────────
#  Step 2's scoring functions all assume the candidate is some kind
#  of engineer, but nothing previously verified that. An HR Manager
#  with long tenure at a product company could still score well on
#  career/tenure/product_ratio alone, because domain match (0%) only
#  cost 35% of the career score, not 100%.
#
#  This gate runs in Step 1 (where we still have the full candidate
#  pool) and EXCLUDES non-technical candidates from the semantic
#  search pool entirely, before any embedding or LLM cost is spent
#  on them. This is a hard filter, not a soft scoring penalty.
# ═══════════════════════════════════════════════════════════════

NON_TECHNICAL_TITLE_SIGNALS = [
    "hr manager", "human resources", "recruiter", "talent acquisition",
    "marketing manager", "marketing executive", "content writer",
    "sales manager", "sales executive", "account manager",
    "business development", "customer success", "customer support",
    "operations manager", "office manager", "administrative",
    "finance manager", "accountant", "legal counsel", "paralegal",
    "graphic designer", "social media", "brand manager",
    "procurement", "supply chain manager", "logistics manager",
]

TECHNICAL_TITLE_SIGNALS = [
    "software engineer", "ml engineer", "machine learning engineer",
    "ai engineer", "data engineer", "platform engineer",
    "backend engineer", "frontend engineer", "full stack engineer",
    "devops engineer", "sre", "infrastructure engineer",
    "nlp engineer", "search engineer", "ranking engineer",
    "developer", "data scientist", "research scientist", "applied scientist",
    "software architect", "solutions architect", "technical architect",
    "programmer", "computer vision engineer",
]

# NOTE: explicitly NON-software engineering disciplines. Without this list,
# the bare substring "engineer" in TECHNICAL_TITLE_SIGNALS would match
# "Civil Engineer" or "Mechanical Engineer" — caught during testing on
# real sample data where these slipped through the gate.
NON_SOFTWARE_ENGINEER_TITLES = [
    "civil engineer", "mechanical engineer", "electrical engineer",
    "chemical engineer", "structural engineer", "industrial engineer",
    "petroleum engineer", "mining engineer", "aerospace engineer",
    "environmental engineer", "biomedical engineer", "agricultural engineer",
]

def is_technical_candidate(c: dict) -> bool:
    """
    Hard gate: True only if there's real evidence this person is a
    software/ML engineer or scientist — not a non-technical professional
    with adjacent keywords, and not a non-software engineering discipline
    (civil, mechanical, etc.) that happens to contain the word "engineer".

    Checks THREE independent signals — any one passing is enough:
      1. Current title contains a technical signal AND no non-technical
         AND no non-software-engineering signal
      2. skill_assessment_scores has entries (platform-administered tests —
         a marketing manager won't have NLP/fine-tuning test results)
      3. At least one job in career_history has a technical title
         (same exclusions applied)
    """
    def _is_tech_title(title: str) -> bool:
        t = title.lower()
        if any(sig in t for sig in NON_SOFTWARE_ENGINEER_TITLES):
            return False
        if any(sig in t for sig in NON_TECHNICAL_TITLE_SIGNALS):
            return False
        return any(sig in t for sig in TECHNICAL_TITLE_SIGNALS)

    p = c.get("profile", {})
    if _is_tech_title(p.get("current_title") or ""):
        return True

    assessed = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if assessed:
        return True

    for job in c.get("career_history", []):
        if _is_tech_title(job.get("title") or ""):
            return True

    return False


# ═══════════════════════════════════════════════════════════════
#  CANDIDATE TEXT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_candidate_text(c: dict) -> str:
    p       = c.get("profile", {})
    history = c.get("career_history", [])
    edu     = c.get("education", [])
    sigs    = c.get("redrob_signals", {})
    parts   = []

    title   = p.get("current_title", "")
    company = p.get("current_company", "")
    yoe     = p.get("years_of_experience", 0)
    if title:
        parts.append(f"{title} at {company}. {yoe} years experience.")
    if p.get("headline"):
        parts.append(p["headline"])

    sorted_hist = sorted(history, key=lambda j: j.get("start_date") or "0000", reverse=True)
    for job in sorted_hist[:5]:
        desc = job.get("description", "")
        role = job.get("title", "")
        co   = job.get("company", "")
        dur  = job.get("duration_months", 0) or 0
        if desc:
            if len(desc) > 300:
                desc = desc[:300].rsplit(" ", 1)[0] + "..."
            parts.append(f"{role} at {co} ({dur}mo): {desc}")

    for e in edu[:1]:
        deg, field = e.get("degree", ""), e.get("field_of_study", "")
        if field:
            parts.append(f"Education: {deg} in {field}.")

    assessed = sigs.get("skill_assessment_scores", {})
    if assessed:
        top = sorted(assessed.items(), key=lambda x: -x[1])[:4]
        parts.append("Assessed: " + ", ".join(f"{k}:{v:.0f}" for k, v in top))

    summary = p.get("summary", "")
    if summary and len(summary) < 400:
        parts.append(summary)

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════
#  RATE-LIMITED GROQ CALLER
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, min_seconds: float):
        self.min_seconds    = min_seconds
        self.last_call_time = 0.0

    def wait(self):
        elapsed   = time.time() - self.last_call_time
        remaining = self.min_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self.last_call_time = time.time()


_rate_limiter = RateLimiter(MIN_SECONDS_BETWEEN_CALLS)


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower()


def get_llm_reason(client, candidate: dict, semantic_score: float) -> str:
    """
    Calls Groq with rate limiting + automatic retry on 429.

    DIVERSITY FIX: the prompt pulls candidate-SPECIFIC anchors — exact
    company name, exact job title, the literal sentence from their job
    description that triggered the match, and their skill-assessment
    results if present. The instruction explicitly forbids generic
    template phrasing like "X years experience" openers, which is what
    caused near-duplicate reasoning across different candidates before.
    """
    p       = candidate.get("profile", {})
    history = candidate.get("career_history", [])

    sorted_hist = sorted(history, key=lambda j: j.get("start_date","0000"), reverse=True)
    recent_roles = [
        f"{j.get('title','')} at {j.get('company','')} ({j.get('duration_months',0)}mo)"
        for j in sorted_hist[:3]
    ]

    # Find the most specific evidence sentence — prefer one with concrete
    # nouns (a system name, a number, a technology) over generic prose.
    candidate_sentences = []
    for j in history:
        d = j.get("description", "")
        for sentence in re.split(r'(?<=[.!?])\s+', d):
            if any(sig in sentence.lower() for sig in [
                "production","deployed","ranking","retrieval","search",
                "recommendation","embedding","vector","a/b","ndcg","mrr",
            ]):
                candidate_sentences.append(sentence.strip())

    evidence = candidate_sentences[0] if candidate_sentences else (
        history[0].get("description","")[:200] if history else "No description available"
    )

    assessed = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    assessed_str = (
        ", ".join(f"{k}: {v:.0f}" for k, v in sorted(assessed.items(), key=lambda x: -x[1])[:3])
        if assessed else "none recorded"
    )

    prompt = f"""You are evaluating ONE specific candidate for a Senior AI Engineer role
building production ranking, retrieval and search systems.

Candidate ID: {candidate.get('candidate_id','')}
Current role: {p.get('current_title','')} at {p.get('current_company','')}
Recent roles: {' → '.join(recent_roles)}
Specific evidence from their work history: "{evidence}"
Platform skill assessments: {assessed_str}
Semantic match score: {semantic_score:.2f}

Write ONE sentence (max 25 words) that is UNIQUE to this candidate.

Rules:
- Quote or closely paraphrase the SPECIFIC evidence sentence above — name the actual
  system, company, or technology mentioned in it.
- Do NOT start with "X years experience" or any generic opener — every candidate gets
  evaluated this way and identical openers make reasoning indistinguishable.
- Do NOT use generic phrases like "strong candidate", "excellent fit", "great experience".
- If the evidence is weak or generic, say so plainly rather than inventing strength."""

    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        _rate_limiter.wait()
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.4,   # slightly higher than 0 — encourages varied phrasing
            )
            return resp.choices[0].message.content.strip().strip('"')

        except Exception as e:
            if _is_rate_limit_error(e):
                if attempt == MAX_RETRIES:
                    return "RATE_LIMITED_AFTER_RETRIES"
                print(f"\n  [429] Rate limited. Retry {attempt}/{MAX_RETRIES} "
                      f"after {backoff}s...", end="")
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                return f"API error: {e}"

    return "RATE_LIMITED_AFTER_RETRIES"


def _bar(i, n, label=""):
    pct  = i / n * 100 if n else 0
    done = int(pct / 2)
    bar  = "█" * done + "░" * (50 - done)
    print(f"\r  [{bar}] {pct:5.1f}%  {label}", end="", flush=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def run():
    t0 = time.time()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # ── Device ─────────────────────────────────────────────────
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = BATCH_SIZE_GPU if device == "cuda" else BATCH_SIZE_CPU
    print(f"\n  Device: {device.upper()}  |  Batch size: {batch_size}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠  No GPU. Runtime → Change runtime type → T4 GPU")

    # ── API key ────────────────────────────────────────────────
    api_key = GROQ_API_KEY
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("GROQ_API_KEY")
            print("  API key: loaded from Colab secrets ✅")
        except Exception:
            api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("  ⚠  No API key found. LLM reasoning will be skipped.")
    else:
        print("  API key: ready ✅")
        print(f"  Model: {GROQ_MODEL}  |  Pacing: 1 call every {MIN_SECONDS_BETWEEN_CALLS}s "
              f"(~{60/MIN_SECONDS_BETWEEN_CALLS:.0f} RPM)")

    # ── Load candidates ────────────────────────────────────────
    print(f"\n  Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        candidates = (json.loads(content) if content.startswith("[")
                      else [json.loads(l) for l in content.splitlines() if l.strip()])
    print(f"  Loaded {len(candidates):,} candidates.")

    # ── Apply technical-role gate ────────────────────────────────
    pre_gate_count = len(candidates)
    candidates = [c for c in candidates if is_technical_candidate(c)]
    excluded_count = pre_gate_count - len(candidates)
    print(f"  Technical-role gate: excluded {excluded_count:,} non-technical "
          f"profiles (e.g. HR, marketing, sales).")
    print(f"  Remaining for semantic search: {len(candidates):,}\n")

    # ── Load model ─────────────────────────────────────────────
    print(f"  Loading model: {MODEL_NAME} (~130MB, cached after first run)")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"  Model loaded on {device.upper()} ✅\n")

    # ── Build texts ────────────────────────────────────────────
    print("  Building candidate texts from career descriptions...")
    texts = [build_candidate_text(c) for c in candidates]
    print(f"  Sample text: {texts[0][:120]}...\n")

    # ── Encode candidates ──────────────────────────────────────
    print(f"  Encoding {len(candidates):,} candidates...")
    prefixed = ["Represent this sentence: " + t for t in texts]
    cand_vecs = model.encode(
        prefixed, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  Done. Shape: {cand_vecs.shape}")

    cand_ids = np.array([c["candidate_id"] for c in candidates])
    emb_path = out / "candidate_embeddings.npz"
    np.savez_compressed(str(emb_path), embeddings=cand_vecs, ids=cand_ids)
    print(f"  Saved: {emb_path}\n")

    # ── Encode JD ──────────────────────────────────────────────
    print("  Encoding 4 JD query facets...")
    query_prefixed = ["Represent this sentence: " + q for q in JD_QUERIES]
    query_vecs = model.encode(
        query_prefixed, normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    jd_path = out / "jd_embedding.npy"
    np.save(str(jd_path), query_vecs)
    print(f"  Saved: {jd_path}  (shape {query_vecs.shape})\n")

    # ── FAISS search ───────────────────────────────────────────
    print("  Running FAISS search (MAX similarity across 4 facets)...")
    if device == "cuda":
        try:
            res       = faiss.StandardGpuResources()
            index_cpu = faiss.IndexFlatIP(EMBEDDING_DIM)
            index     = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            print("  FAISS on GPU ✅")
        except Exception:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            print("  FAISS GPU unavailable, using CPU.")
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)

    index.add(cand_vecs)

    k          = min(TOP_SEMANTIC * 3, len(candidates))
    all_scores = np.full(len(candidates), -1.0, dtype=np.float32)
    for qi, q_vec in enumerate(query_vecs):
        sims, idxs = index.search(q_vec.reshape(1, -1), k)
        for idx, sim in zip(idxs[0], sims[0]):
            if sim > all_scores[idx]:
                all_scores[idx] = sim
        print(f"  Query {qi+1}/4 done. Best sim: {sims[0][0]:.4f}")

    # ── Stable sort for determinism: tie-break by candidate_id ─
    # argsort alone can order equal-score items arbitrarily depending on
    # internal memory layout. We sort by (-score, candidate_id) so reruns
    # on identical input always produce identical top-300 ordering.
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-all_scores[i], candidates[i]["candidate_id"])
    )
    top_idx = order[:TOP_SEMANTIC]
    print(f"\n  Top {TOP_SEMANTIC} retrieved (deterministic order). Score range: "
          f"{all_scores[top_idx[0]]:.4f} – {all_scores[top_idx[-1]]:.4f}\n")

    # ── Resume support ──────────────────────────────────────────
    checkpoint_path = out / "top300_with_reasons.json.checkpoint"
    already_done = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            prior = json.load(f)
        already_done = {c["candidate_id"]: c for c in prior}
        print(f"  Found checkpoint with {len(already_done)} candidates already processed.")
        print(f"  Resuming from where it left off...\n")

    # ── Attach semantic score + LLM reason to top 300 ──────────
    est_minutes = (TOP_SEMANTIC * MIN_SECONDS_BETWEEN_CALLS) / 60
    print(f"  Generating LLM reasoning for top {TOP_SEMANTIC} candidates...")
    print(f"  Estimated time at current pacing: ~{est_minutes:.1f} minutes\n")

    client = Groq(api_key=api_key) if api_key else None

    top300 = []
    rate_limited_count = 0
    for i, idx in enumerate(top_idx):
        c = dict(candidates[idx])
        cid = c["candidate_id"]
        sem_score = float(all_scores[idx])
        c["_semantic_score"] = round(sem_score, 4)

        if cid in already_done and already_done[cid].get("_llm_reason") not in (
            None, "", "RATE_LIMITED_AFTER_RETRIES"
        ):
            c["_llm_reason"] = already_done[cid]["_llm_reason"]
        elif client:
            c["_llm_reason"] = get_llm_reason(client, c, sem_score)
            if c["_llm_reason"] == "RATE_LIMITED_AFTER_RETRIES":
                rate_limited_count += 1
        else:
            c["_llm_reason"] = "LLM reasoning skipped — no API key"

        top300.append(c)

        if (i + 1) % 10 == 0 or (i + 1) == TOP_SEMANTIC:
            _bar(i + 1, TOP_SEMANTIC,
                 f"{i+1}/{TOP_SEMANTIC}  (rate-limited so far: {rate_limited_count})")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(str(checkpoint_path), "w", encoding="utf-8") as f:
                json.dump(top300, f, ensure_ascii=False)
    print()

    if rate_limited_count > 0:
        print(f"\n  ⚠  {rate_limited_count} candidates still hit rate limits after "
              f"{MAX_RETRIES} retries each. Re-run this script to resume.")

    # ── Diversity check — warn if reasoning text repeats ────────
    reasons = [c.get("_llm_reason","") for c in top300]
    unique_reasons = len(set(reasons))
    print(f"\n  Diversity check: {unique_reasons}/{len(reasons)} unique reasoning sentences.")
    if unique_reasons < len(reasons) * 0.9:
        print("  ⚠  Significant repetition detected — review prompt or evidence extraction.")

    # ── Save ──────────────────────────────────────────────────────
    top300_path = out / "top300_with_reasons.json"
    with open(str(top300_path), "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, c in enumerate(top300):
            comma = "," if i < len(top300) - 1 else ""
            f.write("  " + json.dumps(c, ensure_ascii=False) + comma + "\n")
        f.write("]\n")
    print(f"\n  Saved: {top300_path}")

    if rate_limited_count == 0 and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  Checkpoint cleared (all candidates processed successfully).")

    elapsed = time.time() - t0
    print(f"""
╔══════════════════════════════════════════════════════════════╗
  PRECOMPUTE COMPLETE  ({elapsed:.0f}s on {device.upper()})
╠══════════════════════════════════════════════════════════════╣
  Input candidates        : {pre_gate_count:,}
  Excluded (non-technical) : {excluded_count:,}
  Technical pool          : {len(candidates):,}
  Top semantic pool        : {TOP_SEMANTIC}
  LLM reasons generated     : {TOP_SEMANTIC - rate_limited_count}
  Unique reasoning lines    : {unique_reasons}/{TOP_SEMANTIC}
  Still rate-limited        : {rate_limited_count}
  ────────────────────────────────────────────────────────────
  Output files (all in {out}):
    jd_embedding.npy
    candidate_embeddings.npz
    top300_with_reasons.json
  ────────────────────────────────────────────────────────────
  NEXT STEP: run step2_score_and_export.py
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    run()