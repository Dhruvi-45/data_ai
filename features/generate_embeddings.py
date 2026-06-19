# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — PRECOMPUTE  (run ONCE in Colab, GPU runtime, no time limit)   ║
# ║                                                                          ║
# ║  This script does ALL the expensive work:                                ║
# ║    1. Embeds the JD (4 query facets)                                     ║
# ║    2. Embeds all ~20k candidates                                         ║
# ║    3. Runs semantic search -> top 300 candidates                         ║
# ║    4. Calls Gemini API for a one-line reason per top-300 candidate       ║
# ║                                                                          ║
# ║  This is run SEPARATELY from your final 5-min CPU-only submission run.   ║
# ║  Run this whenever you want, on GPU, and just save the outputs.          ║
# ║                                                                          ║
# ║  OUTPUTS (saved to OUTPUT_DIR):                                          ║
# ║    jd_embedding.npy              — JD vectors (4 facets × 384 dims)      ║
# ║    candidate_embeddings.npz      — all candidate vectors + IDs           ║
# ║    top300_with_reasons.json      — top 300 candidates, with:             ║
# ║                                     - semantic_score (precomputed)       ║
# ║                                     - llm_reason (precomputed)           ║
# ║                                     - full original candidate data       ║
# ║                                                                          ║
# ║  STEP 0: Runtime → Change runtime type → T4 GPU                          ║
# ║  STEP 1: Mount Drive:                                                    ║
# ║            from google.colab import drive                                ║
# ║            drive.mount('/content/drive')                                 ║
# ║  STEP 2: Install:                                                        ║
# ║            !pip install sentence-transformers faiss-gpu google-genai --quiet
# ║            # if faiss-gpu fails: !pip install faiss-cpu google-genai --quiet
# ║  STEP 3: Set GEMINI_API_KEY in Colab Secrets (🔑 icon) or below         ║
# ║  STEP 4: Update INPUT_PATH / OUTPUT_DIR, then Run All                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os, json, time
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
    from google import genai
    from google.genai import types
except ImportError:
    raise SystemExit("Run: !pip install google-genai --quiet")
import torch

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

INPUT_PATH  = "againhoneyfiltered_candidates (1).json"   # your ~20k filtered file
OUTPUT_DIR  = "/content/drive/MyDrive/redrob/outputs"

GEMINI_API_KEY = ""   # paste here, or leave blank to use Colab Secrets

MODEL_NAME     = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM  = 384
BATCH_SIZE_GPU = 512
BATCH_SIZE_CPU = 128

TOP_SEMANTIC = 300   # how many candidates to carry forward to scoring stage

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
#  CANDIDATE TEXT BUILDER
#  Uses career DESCRIPTIONS (what they DID), NOT skills[] (keyword soup).
#  This is the primary anti-keyword-trap move.
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
#  GEMINI REASONING — one crisp sentence per candidate
# ═══════════════════════════════════════════════════════════════

def get_llm_reason(client, candidate: dict, semantic_score: float) -> str:
    p       = candidate.get("profile", {})
    history = candidate.get("career_history", [])

    recent_roles = [
        f"{j.get('title','')} at {j.get('company','')} ({j.get('duration_months',0)}mo)"
        for j in sorted(history, key=lambda j: j.get("start_date","0000"), reverse=True)[:3]
    ]
    top_desc = ""
    for j in history:
        d = j.get("description", "")
        if any(sig in d.lower() for sig in ["production","deployed","ranking","retrieval","search"]):
            top_desc = d[:250]
            break

    prompt = f"""You are evaluating candidates for a Senior AI Engineer role building
production ranking, retrieval and search systems (real shipped systems, not research).

Candidate: {p.get('current_title','')} at {p.get('current_company','')}, {p.get('years_of_experience',0)}y exp
Recent: {' → '.join(recent_roles)}
Key work: {top_desc or 'N/A'}
Semantic match score: {semantic_score:.2f}

Write ONE sentence (max 25 words). State the most specific concrete reason this candidate
fits this role — name the actual system or technology they shipped. No generic phrases
like "strong candidate" or "excellent fit"."""

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=80,
                temperature=0.2
            )
        )
        return resp.text.strip().strip('"')
    except Exception as e:
        return f"API error: {e}"


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
        print("  ⚠  No GPU. This script has no time limit, but GPU is much faster.")
        print("     Runtime → Change runtime type → T4 GPU")

    # ── API key ────────────────────────────────────────────────
    api_key = GEMINI_API_KEY
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("GEMINI_API_KEY")
            print("  API key: loaded from Colab secrets ✅")
        except Exception:
            api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("  ⚠  No API key found. LLM reasoning will be skipped (filled with placeholder).")
    else:
        print("  API key: ready ✅")

    # ── Load candidates ────────────────────────────────────────
    print(f"\n  Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        candidates = (json.loads(content) if content.startswith("[")
                      else [json.loads(l) for l in content.splitlines() if l.strip()])
    print(f"  Loaded {len(candidates):,} candidates.\n")

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
        sim, idxs = index.search(q_vec.reshape(1, -1), k)
        for idx, sim in zip(idxs[0], sim[0]):
            if sim > all_scores[idx]:
                all_scores[idx] = sim
        print(f"  Query {qi+1}/4 done. Best sim: {sim[0][0]:.4f}")

    top_idx = np.argsort(all_scores)[::-1][:TOP_SEMANTIC]
    print(f"\n  Top {TOP_SEMANTIC} retrieved. Score range: "
          f"{all_scores[top_idx[0]]:.4f} – {all_scores[top_idx[-1]]:.4f}\n")

    # ── Attach semantic score + LLM reason to top 300 ──────────
    print(f"  Generating LLM reasoning for top {TOP_SEMANTIC} candidates...")
    client = genai.Client(api_key=api_key) if api_key else None

    top300 = []
    for i, idx in enumerate(top_idx):
        c = dict(candidates[idx])   # copy — don't mutate original
        sem_score = float(all_scores[idx])
        c["_semantic_score"] = round(sem_score, 4)

        if client:
            c["_llm_reason"] = get_llm_reason(client, c, sem_score)
            time.sleep(0.1)   # gentle rate-limit buffer
        else:
            c["_llm_reason"] = "LLM reasoning skipped — no API key"

        top300.append(c)
        if (i + 1) % 10 == 0 or (i + 1) == TOP_SEMANTIC:
            _bar(i + 1, TOP_SEMANTIC, f"{i+1}/{TOP_SEMANTIC}")
    print()

    # ── Save top300 with reasons ────────────────────────────────
    top300_path = out / "top300_with_reasons.json"
    with open(str(top300_path), "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, c in enumerate(top300):
            comma = "," if i < len(top300) - 1 else ""
            f.write("  " + json.dumps(c, ensure_ascii=False) + comma + "\n")
        f.write("]\n")
    print(f"\n  Saved: {top300_path}")

    elapsed = time.time() - t0
    print(f"""
╔══════════════════════════════════════════════════════════════╗
  PRECOMPUTE COMPLETE  ({elapsed:.0f}s on {device.upper()})
╠══════════════════════════════════════════════════════════════╣
  Input candidates  : {len(candidates):,}
  Top semantic pool : {TOP_SEMANTIC}
  ────────────────────────────────────────────────────────────
  Output files (all in {out}):
    jd_embedding.npy            — 4 JD facet vectors
    candidate_embeddings.npz    — all {len(candidates):,} candidate vectors
    top300_with_reasons.json    — top {TOP_SEMANTIC}, with semantic score + LLM reason
  ────────────────────────────────────────────────────────────
  NEXT STEP:
  Run step2_score_and_export.py — it loads top300_with_reasons.json
  only, applies redrob_signals scoring, and writes top100_final.csv.
  That script has NO model loading, NO embedding, NO API calls —
  it is pure numpy/python and will finish in seconds on CPU.
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    run()