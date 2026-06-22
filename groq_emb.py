# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — PRECOMPUTE  (Groq, de-duplicated LLM reasons)                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os, json, time, re
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("Run: !pip install sentence-transformers --quiet")
try:
    import faiss
except ImportError:
    raise SystemExit("Run: !pip install faiss-gpu-cu12 --quiet")
try:
    from groq import Groq
except ImportError:
    raise SystemExit("Run: !pip install groq --quiet")
import torch

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]        = "1"

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

INPUT_PATH  = "againhoneyfiltered_candidates (1).json"
OUTPUT_DIR  = "/content/drive/MyDrive/redrob/outputs"

GROQ_API_KEY = ""        # paste here or use Colab Secrets
GROQ_MODEL   = "llama-3.1-8b-instant"   # or "llama-3.3-70b-versatile"

MODEL_NAME     = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM  = 384
BATCH_SIZE_GPU = 512
BATCH_SIZE_CPU = 128

TOP_SEMANTIC = 300

MIN_SECONDS_BETWEEN_CALLS = 0.2
MAX_RETRIES               = 4
INITIAL_BACKOFF_SECONDS   = 4
CHECKPOINT_EVERY          = 20

# Similarity threshold — reasons with >75% overlap trigger a re-call
DEDUP_SIMILARITY_THRESHOLD = 0.75
# Max re-call attempts to fix a duplicate before accepting it
DEDUP_MAX_ATTEMPTS = 3

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


def extract_candidate_signals(c: dict) -> dict:
    """
    Pull the most unique/specific facts about this candidate so we can
    inject them directly into the prompt. This is the main fix for
    repeated generic reasons — the model is forced to reference real data.
    """
    p       = c.get("profile", {})
    history = c.get("career_history", [])
    edu     = c.get("education", [])
    sigs    = c.get("redrob_signals", {})

    # Most recent 3 job titles + companies
    sorted_hist = sorted(history, key=lambda j: j.get("start_date") or "0000", reverse=True)
    recent_roles = [
        f"{j.get('title','')} at {j.get('company','')} ({j.get('duration_months') or 0}mo)"
        for j in sorted_hist[:3]
    ]

    # First job description that mentions a production/technical keyword
    # Try progressively broader keywords so we always find something
    top_desc  = ""
    top_role  = ""
    top_co    = ""
    kw_tiers  = [
        ["production", "deployed", "ranking", "retrieval", "search", "recommendation"],
        ["built", "developed", "designed", "led", "architected", "scaled"],
        ["model", "pipeline", "system", "api", "service", "infrastructure"],
    ]
    for keywords in kw_tiers:
        for j in sorted_hist:
            d = j.get("description", "")
            if any(kw in d.lower() for kw in keywords):
                top_desc = d[:300].rsplit(" ", 1)[0]
                top_role = j.get("title", "")
                top_co   = j.get("company", "")
                break
        if top_desc:
            break

    # Skills (assessed first, then listed)
    assessed = sigs.get("skill_assessment_scores", {})
    top_assessed = sorted(assessed.items(), key=lambda x: -x[1])[:3] if assessed else []
    listed_skills = p.get("skills", [])[:6] if p.get("skills") else []

    # Education
    edu_str = ""
    if edu:
        e = edu[0]
        edu_str = f"{e.get('degree','')} in {e.get('field_of_study','')} from {e.get('school','')}"

    # RedRob-specific signals
    prod_signals = sigs.get("production_system_signals", [])
    noteworthy   = sigs.get("noteworthy_employers", [])

    return {
        "name":           p.get("name") or p.get("full_name") or "Candidate",
        "title":          p.get("current_title", ""),
        "company":        p.get("current_company", ""),
        "yoe":            p.get("years_of_experience", 0),
        "location":       p.get("location", ""),
        "recent_roles":   recent_roles,
        "top_desc":       top_desc,
        "top_role":       top_role,
        "top_co":         top_co,
        "assessed":       ", ".join(f"{k}({v:.0f})" for k, v in top_assessed),
        "skills":         ", ".join(listed_skills),
        "education":      edu_str,
        "prod_signals":   ", ".join(prod_signals[:3]) if prod_signals else "",
        "noteworthy_cos": ", ".join(noteworthy[:3]) if noteworthy else "",
        "headline":       p.get("headline", ""),
    }


# ═══════════════════════════════════════════════════════════════
#  RATE LIMITER
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


# ═══════════════════════════════════════════════════════════════
#  DUPLICATE DETECTOR
# ═══════════════════════════════════════════════════════════════

def _similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio — fast, no deps, good enough for short sentences."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def is_duplicate(reason: str, seen_reasons: list[str], threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> bool:
    """Returns True if `reason` is too similar to any previously accepted reason."""
    if not reason or reason.startswith(("API error", "RATE_LIMITED", "Skipped")):
        return False   # don't flag error strings as duplicates
    for prev in seen_reasons:
        if _similarity(reason, prev) >= threshold:
            return True
    return False


# ═══════════════════════════════════════════════════════════════
#  GROQ LLM CALLER  — rich prompt, de-duplicate aware
# ═══════════════════════════════════════════════════════════════

def build_prompt(sig: dict, semantic_score: float, candidate_index: int,
                 avoid_phrases: list[str] | None = None) -> str:
    """
    Builds a prompt packed with candidate-specific facts so the model
    is forced to write something unique every time.
    `avoid_phrases` is injected when we're retrying a duplicate.
    """
    avoid_block = ""
    if avoid_phrases:
        quoted = "; ".join(f'"{p}"' for p in avoid_phrases[-3:])
        avoid_block = (
            f"\n\nIMPORTANT: The following reason(s) were already used for other candidates. "
            f"Do NOT write anything similar to: {quoted}. "
            f"Find a DIFFERENT specific angle for this person."
        )

    skills_block = ""
    if sig["assessed"]:
        skills_block += f"\nAssessed skills: {sig['assessed']}"
    if sig["skills"]:
        skills_block += f"\nListed skills: {sig['skills']}"
    if sig["prod_signals"]:
        skills_block += f"\nProduction signals: {sig['prod_signals']}"
    if sig["noteworthy_cos"]:
        skills_block += f"\nNotable employers: {sig['noteworthy_cos']}"

    return f"""You are evaluating candidate #{candidate_index} for a Senior AI Engineer role.
The role requires shipping REAL production ranking, retrieval, and search systems to actual users.

━━ CANDIDATE FACTS (use these — do not invent) ━━
Name/ID       : {sig['name']} (#{candidate_index})
Current role  : {sig['title']} at {sig['company']}
Experience    : {sig['yoe']} years | Location: {sig['location']}
Headline      : {sig['headline'] or 'N/A'}
Recent path   : {' → '.join(sig['recent_roles']) or 'N/A'}
Education     : {sig['education'] or 'N/A'}{skills_block}

Most relevant work ({sig['top_role']} at {sig['top_co']}):
  {sig['top_desc'] or 'No production-keyword description found — use skills/role instead.'}

Semantic relevance score: {semantic_score:.3f} / 1.000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Write exactly ONE sentence, max 25 words.
- Name the SPECIFIC technology, system, or company that makes this person unique.
- Ground it in the facts above — no invention.
- No filler like "strong candidate", "excellent fit", "proven track record".
- If their background is weak for the role, say what they DO have concretely.
- Output ONLY the sentence. No preamble, no quotes, no period at start.{avoid_block}"""


def get_llm_reason(
    client: Groq,
    candidate: dict,
    semantic_score: float,
    candidate_index: int,
    seen_reasons: list[str],
) -> str:
    """
    Calls Groq with a rich, candidate-specific prompt.
    If the returned reason is a near-duplicate of a previous one,
    retries up to DEDUP_MAX_ATTEMPTS times with an explicit avoid list.
    """
    sig          = extract_candidate_signals(candidate)
    avoid        = []          # phrases to avoid on retry
    final_reason = "RATE_LIMITED_AFTER_RETRIES"

    for dedup_attempt in range(DEDUP_MAX_ATTEMPTS):
        prompt  = build_prompt(sig, semantic_score, candidate_index,
                               avoid_phrases=avoid if dedup_attempt > 0 else None)
        backoff = INITIAL_BACKOFF_SECONDS

        # ── inner retry loop (for 429s only) ──────────────────
        for api_attempt in range(1, MAX_RETRIES + 1):
            _rate_limiter.wait()
            try:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You write precise, fact-grounded one-sentence candidate evaluations. "
                                "Each sentence must be unique — never repeat phrasing used for other candidates. "
                                "Output ONLY the sentence, nothing else."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=70,
                    temperature=0.45 + (dedup_attempt * 0.1),  # increase variety on retry
                )
                reason = resp.choices[0].message.content.strip().strip('"').strip("'")

                # ── duplicate check ────────────────────────────
                if is_duplicate(reason, seen_reasons):
                    avoid.append(reason)   # tell the model to avoid this on next attempt
                    print(f"\n  [dedup] Attempt {dedup_attempt+1}: duplicate detected, retrying...", end="")
                    break   # break inner loop → try outer dedup_attempt loop again
                else:
                    return reason   # ✅ unique reason accepted

            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate_limit" in msg.lower():
                    if api_attempt == MAX_RETRIES:
                        return "RATE_LIMITED_AFTER_RETRIES"
                    print(f"\n  [429] Backing off {backoff}s (attempt {api_attempt}/{MAX_RETRIES})...", end="")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    return f"API error: {e}"
        else:
            # inner loop exhausted retries without success (only happens on 429 cascade)
            return final_reason

    # All dedup attempts exhausted — return last generated reason (better than nothing)
    return avoid[-1] if avoid else final_reason


def _bar(i, n, label=""):
    pct  = i / n * 100 if n else 0
    done = int(pct / 2)
    bar  = "█" * done + "░" * (50 - done)
    print(f"\r  [{bar}] {pct:5.1f}%  {label}", end="", flush=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def run():
    t0  = time.time()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # ── Device ─────────────────────────────────────────────────
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = BATCH_SIZE_GPU if device == "cuda" else BATCH_SIZE_CPU
    print(f"\n  Device: {device.upper()}  |  Batch size: {batch_size}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── API key ────────────────────────────────────────────────
    api_key = GROQ_API_KEY
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("GROQ_API_KEY")
            print("  Groq API key: loaded from Colab secrets ✅")
        except Exception:
            api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("  ⚠  No Groq API key. LLM reasoning will be skipped.")
        client = None
    else:
        print(f"  Groq API key: ready ✅  |  Model: {GROQ_MODEL}")
        client = Groq(api_key=api_key)

    # ── Load candidates ────────────────────────────────────────
    print(f"\n  Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        candidates = (json.loads(content) if content.startswith("[")
                      else [json.loads(l) for l in content.splitlines() if l.strip()])
    print(f"  Loaded {len(candidates):,} candidates.\n")

    # ── Load embedding model ───────────────────────────────────
    print(f"  Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"  Model loaded on {device.upper()} ✅\n")

    # ── Build texts + encode ───────────────────────────────────
    print("  Building candidate texts from career descriptions...")
    texts    = [build_candidate_text(c) for c in candidates]
    prefixed = ["Represent this sentence: " + t for t in texts]

    print(f"  Encoding {len(candidates):,} candidates...")
    cand_vecs = model.encode(
        prefixed, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)

    cand_ids = np.array([c["candidate_id"] for c in candidates])
    np.savez_compressed(str(out / "candidate_embeddings.npz"), embeddings=cand_vecs, ids=cand_ids)
    print(f"  Candidate embeddings saved.\n")

    # ── Encode JD facets ───────────────────────────────────────
    print("  Encoding 4 JD query facets...")
    query_prefixed = ["Represent this sentence: " + q for q in JD_QUERIES]
    query_vecs = model.encode(
        query_prefixed, normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    np.save(str(out / "jd_embedding.npy"), query_vecs)
    print(f"  JD embeddings saved.\n")

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

    top_idx = np.argsort(all_scores)[::-1][:TOP_SEMANTIC]
    print(f"\n  Top {TOP_SEMANTIC} retrieved. Score range: "
          f"{all_scores[top_idx[0]]:.4f} – {all_scores[top_idx[-1]]:.4f}\n")

    # ── Resume from checkpoint ─────────────────────────────────
    checkpoint_path = out / "top300_with_reasons.json.checkpoint"
    already_done    = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            prior = json.load(f)
        already_done = {c["candidate_id"]: c for c in prior}
        print(f"  Checkpoint found: {len(already_done)} candidates already processed. Resuming...\n")

    # ── LLM reasoning loop ─────────────────────────────────────
    print(f"  Generating de-duplicated LLM reasons for top {TOP_SEMANTIC} candidates...\n")

    top300             = []
    seen_reasons       = []   # tracks all accepted reasons for duplicate detection
    rate_limited_count = 0
    dedup_retries      = 0

    for i, idx in enumerate(top_idx):
        c         = dict(candidates[idx])
        cid       = c["candidate_id"]
        sem_score = float(all_scores[idx])
        c["_semantic_score"] = round(sem_score, 4)

        # Use checkpoint if already successfully processed
        if cid in already_done and already_done[cid].get("_llm_reason") not in (
            None, "", "RATE_LIMITED_AFTER_RETRIES"
        ):
            reason = already_done[cid]["_llm_reason"]
            c["_llm_reason"] = reason
            if not is_duplicate(reason, seen_reasons):
                seen_reasons.append(reason)

        elif client:
            reason = get_llm_reason(
                client, c, sem_score,
                candidate_index=i + 1,
                seen_reasons=seen_reasons,
            )
            c["_llm_reason"] = reason
            if reason == "RATE_LIMITED_AFTER_RETRIES":
                rate_limited_count += 1
            elif not reason.startswith("API error"):
                seen_reasons.append(reason)

        else:
            c["_llm_reason"] = "Skipped — No Groq API key"

        top300.append(c)

        if (i + 1) % 10 == 0 or (i + 1) == TOP_SEMANTIC:
            _bar(i + 1, TOP_SEMANTIC,
                 f"{i+1}/{TOP_SEMANTIC}  rate-limited: {rate_limited_count}")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(str(checkpoint_path), "w", encoding="utf-8") as f:
                json.dump(top300, f, ensure_ascii=False)

    print()

    # ── Post-process: count actual duplicates that slipped through ─
    final_reasons    = [c["_llm_reason"] for c in top300
                        if c["_llm_reason"] and not c["_llm_reason"].startswith(("API", "RATE", "Skip"))]
    seen_check       = []
    duplicate_count  = 0
    for r in final_reasons:
        if is_duplicate(r, seen_check):
            duplicate_count += 1
        else:
            seen_check.append(r)

    # ── Save final output ──────────────────────────────────────
    top300_path = out / "top300_with_reasons.json"
    with open(str(top300_path), "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, entry in enumerate(top300):
            comma = "," if i < len(top300) - 1 else ""
            f.write("  " + json.dumps(entry, ensure_ascii=False) + comma + "\n")
        f.write("]\n")
    print(f"\n  Saved: {top300_path}")

    if rate_limited_count == 0 and checkpoint_path.exists():
        checkpoint_path.unlink()
        print("  Checkpoint cleared.")

    elapsed = time.time() - t0
    print(f"""
╔══════════════════════════════════════════════════════════════╗
  PRECOMPUTE COMPLETE  ({elapsed:.0f}s on {device.upper()})
╠══════════════════════════════════════════════════════════════╣
  Input candidates         : {len(candidates):,}
  Top semantic pool        : {TOP_SEMANTIC}
  LLM reasons generated    : {TOP_SEMANTIC - rate_limited_count}
  Still rate-limited       : {rate_limited_count}
  Duplicate reasons (final): {duplicate_count}
  ────────────────────────────────────────────────────────────
  Output files in: {out}
    jd_embedding.npy
    candidate_embeddings.npz
    top300_with_reasons.json
  ────────────────────────────────────────────────────────────
  NEXT STEP: Run step2_score_and_export.py
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    run()