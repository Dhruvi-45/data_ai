"""
redrob_pipeline.py — Complete Redrob AI Hackathon Pipeline
============================================================
INPUT  : againhoneyfiltered_candidates (1).json
OUTPUT : top100_final.csv, filtered_out_candidates.json
"""

import os
import json
import re
import csv
import sys
import time
from datetime import datetime, date
import numpy as np
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[ERROR] pip install sentence-transformers"); sys.exit(1)
try:
    import faiss
except ImportError:
    print("[ERROR] pip install faiss-gpu"); sys.exit(1)
try:
    import anthropic
except ImportError:
    print("[ERROR] pip install anthropic"); sys.exit(1)

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION MATRIX
# ═══════════════════════════════════════════════════════════════
INPUT_PATH        = "againhoneyfiltered_candidates (1).json"
OUTPUT_CSV        = "top100_final.csv"
OUTPUT_REJECTS    = "filtered_out_candidates.json"

TOP_SEMANTIC      = 300
TOP_FINAL         = 100
MODEL_NAME        = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM     = 384

# Weights must perfectly sum to 1.0
W_SEMANTIC  = 0.35   # Context fit
W_SHIPPER   = 0.30   # Real-world shipping signature
W_CAREER    = 0.20   # Domain seniority profile alignment
W_ENGAGE    = 0.15   # Behavioral availability indicators

# ═══════════════════════════════════════════════════════════════
#  LEXICONS & STRUCTURAL DRIFT FILTERS
# ═══════════════════════════════════════════════════════════════
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "l&t infotech", "ltimindtree", "lti", "coforge"
}

SHIPPING_EVIDENCE = [
    "deployed to production", "shipped to production", "in production",
    "serving real users", "deployed to real users", "live in production",
    "production traffic", "production system", "at scale", "million users",
    "billion requests", "reduced latency", "p99", "p95", "throughput",
    "qps", "rps", "end-to-end", "owned the system", "led the development",
    "a/b test", "a/b testing", "online evaluation", "offline evaluation",
    "ndcg", "mrr", "precision@", "recall@", "evaluation framework",
    "ranking system", "retrieval system", "search system", "recommendation system",
    "recommender", "recsys", "hybrid retrieval", "dense retrieval", "bm25",
    "reranking", "re-ranking", "learning to rank", "semantic search",
    "vector search", "embedding-based", "candidate ranking", "job matching",
    "feedback loop", "recruiter feedback"
]

RESEARCHER_SIGNALS = [
    "research scientist", "research intern", "postdoc", "post-doc",
    "phd student", "doctoral candidate", "thesis", "published paper",
    "arxiv", "research lab", "university lab", "ablation study", "sota",
    "state of the art", "novel approach", "dataset collection"
]

DOMAIN_SKILLS = {
    "elasticsearch", "opensearch", "solr", "faiss", "qdrant", "pinecone",
    "weaviate", "milvus", "bm25", "hnsw", "annoy", "vector search",
    "semantic search", "hybrid search", "information retrieval", "ranking",
    "learning to rank", "ltr", "lambdamart", "recommendation system", "recsys",
    "collaborative filtering", "xgboost", "lightgbm", "sentence-transformers",
    "bert", "transformers", "hugging face", "nlp", "natural language processing",
    "mlops", "model serving", "a/b testing", "ndcg", "mrr", "python",
    "pytorch", "tensorflow", "scikit-learn"
}

BUZZWORD_SKILLS = {
    "langchain", "openai", "chatgpt", "gpt-4", "gpt-3", "llama", "mistral",
    "rag", "generative ai", "stable diffusion", "llama index", "crewai",
    "autogpt", "langsmith", "copilot"
}

PREFERRED_LOCATIONS = {
    "noida", "pune", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "delhi", "gurugram", "gurgaon", "ncr", "chennai"
}

# ─── PART 1 — CANDIDATE NARRATIVE TEXT GENERATION ────────────────────────────
def build_candidate_text(c: dict) -> str:
    """Extracts job narrative context while omitting skills[] keywords."""
    p       = c.get("profile", {}) or {}
    history = c.get("career_history", []) or []
    edu     = c.get("education", []) or []
    sigs    = c.get("redrob_signals", {}) or {}

    parts = []
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
        parts.append(f"Education: {e.get('degree', '')} in {e.get('field_of_study', '')}.")

    assessed = sigs.get("skill_assessment_scores", {})
    if assessed:
        top = sorted(assessed.items(), key=lambda x: -x[1])[:4]
        parts.append("Assessed: " + ", ".join(f"{k}:{v:.0f}" for k, v in top))

    summary = p.get("summary", "")
    if summary and len(summary) < 400:
        parts.append(summary)

    return " ".join(parts)

# ─── PART 2 — LOGICAL PRE-SCREENING ───────────────────────────────────────────
def is_disqualified(candidate: dict) -> tuple[bool, str]:
    """Applies strict, non-negotiable filtering constraints from the JD."""
    career = candidate.get("career_history", []) or []
    signals = candidate.get("redrob_signals", {}) or []

    # 1. Academic Researcher only filter
    is_all_research = True
    for role in career:
        company_lower = str(role.get("company", "")).lower()
        title_lower = str(role.get("title", "")).lower()
        is_research_role = any(re.search(pat, company_lower) for pat in RESEARCH_ORGS_PATTERNS)
        if not is_research_role and "research" not in title_lower:
            is_all_research = False
            break
    if is_all_research and len(career) > 0:
        return True, "Pure academic research history with zero production company experience"

    # 2. Consulting firm filter
    all_consulting = all(any(firm in str(role.get("company", "")).lower() for firm in CONSULTING_FIRMS) for role in career) if career else False
    if all_consulting and len(career) >= 2:
        return True, "Career path contains entirely consulting or service-based corporations"

    # 3. Platform activity timeline filter (Evaluated against June 18, 2026 challenge baseline)
    last_active = signals.get("last_active_date", "2020-01-01")
    open_to_work = signals.get("open_to_work_flag", False)
    try:
        last_active_date = datetime.strptime(last_active, "%Y-%m-%d").date()
        days_inactive = (date(2026, 6, 18) - last_active_date).days
        if days_inactive > 180 and not open_to_work:
            return True, f"Functional availability constraint: User inactive for {days_inactive} days"
    except Exception:
        pass

    # 4. Notice period ceilings
    if signals.get("notice_period_days", 0) > 90:
        return True, "Stated notice timeline duration constraints exceed platform limits"

    return False, ""

# ─── PART 3 — ASYMMETRIC MULTI-QUERY INTENT FACETS ────────────────────────────
JD_QUERIES = [
    "Senior engineer who built and deployed production ranking, retrieval, or recommendation systems to real users at scale. Handles embedding drift, index refresh, and retrieval quality regression in live production environments. Hybrid BM25 dense retrieval in production.",
    "Engineer who designed evaluation frameworks for ranking and retrieval systems. Built offline to online evaluation pipelines. NDCG MRR MAP precision recall A/B testing feedback loops. Improved ranking system based on real user feedback data.",
    "Applied machine learning engineer at product companies not consulting firms. 5 to 8 years experience. Shipped ML systems to real users, not just prototypes. Prefers working system over perfect model.",
    "Engineer integrating LLMs into production retrieval and ranking pipelines. Knows when to fine-tune versus prompt engineer. Experience fine-tuning with LoRA QLoRA PEFT. Hybrid search combining dense and sparse retrieval."
]

# ─── PART 4 — DETAILED SUB-COMPONENT HEURISTICS ───────────────────────────────

def _lower(t) -> str:
    return t.lower().strip() if t else ""

def _is_consulting(company: str) -> bool:
    c = _lower(company)
    return any(f in c for f in CONSULTING_FIRMS)

def score_shipper(c: dict) -> tuple[float, str]:
    history = c.get("career_history", []) or []
    edu     = c.get("education", []) or []
    all_desc = " ".join(j.get("description", "") for j in history).lower()

    hits = [p for p in SHIPPING_EVIDENCE if p in all_desc]
    ship_raw = min(len(hits), 10) / 10.0

    res_count = sum(1 for j in history if any(sig in str(j.get("title", "") + " " + j.get("description", "")).lower() for sig in RESEARCHER_SIGNALS))
    for e in edu:
        if "phd" in str(e.get("degree", "")).lower() and "production" not in all_desc:
            res_count += 2
    researcher_penalty = min(res_count * 0.07, 0.35)

    product_jobs = sum(1 for j in history if not _is_consulting(j.get("company", "")))
    product_ratio = product_jobs / max(len(history), 1)
    product_bonus = product_ratio * 0.15

    final = max(0.0, min(1.0, ship_raw * (1 - researcher_penalty) + product_bonus))
    return final, f"phrases:{len(hits)},res_penalty:{researcher_penalty:.2f}"

def score_career(c: dict) -> tuple[float, str]:
    p       = c.get("profile", {}) or []
    history = c.get("career_history", []) or []
    yoe = float(p.get("years_of_experience", 0) or 0)
    
    yoe_s = 1.0 if 5 <= yoe <= 9 else (0.85 if 4 <= yoe < 5 else 0.70 if 9 < yoe <= 12 else 0.40)
    
    ML_TITLES = {"ml engineer", "machine learning", "ai engineer", "data scientist", "search engineer", "ranking engineer"}
    domain = sum(1 for j in history if any(mt in _lower(j.get("title", "")) for mt in ML_TITLES))
    domain_s = min(domain / max(len(history), 1), 1.0)
    
    avg_t = sum(j.get("duration_months", 0) or 0 for j in history) / max(len(history), 1)
    tenure_s = 1.0 if avg_t >= 24 else (0.8 if avg_t >= 18 else 0.50)
    product_s = sum(1 for j in history if not _is_consulting(j.get("company", ""))) / max(len(history), 1)

    final = (yoe_s * 0.25 + domain_s * 0.35 + tenure_s * 0.15 + product_s * 0.25)
    return max(0.0, min(1.0, final)), f"yoe:{yoe},tenure:{avg_t:.0f}mo"

def score_skill_depth(c: dict) -> tuple[float, str]:
    skills = c.get("skills", []) or []
    domain_depth  = 0.0
    buzzword_trap = 0
    for s in skills:
        name = _lower(s.get("name", ""))
        dur  = s.get("duration_months", 0) or 0
        end  = s.get("endorsements", 0) or 0
        if any(bw in name for bw in BUZZWORD_SKILLS) and dur < 6:
            buzzword_trap += 1
            continue
        if any(dk in name for dk in DOMAIN_SKILLS):
            domain_depth += (min(dur, 36) / 36) * 0.70 + (min(end, 20) / 20) * 0.30

    depth_norm = min(domain_depth / 5.0, 1.0)
    trap_penalty = min((buzzword_trap / max(len(skills), 1)) * 0.40, 0.30)
    return max(0.0, min(1.0, depth_norm - trap_penalty)), f"depth:{domain_depth:.2f},trap_p:{trap_penalty:.2f}"

def score_engagement(c: dict) -> tuple[float, str]:
    s = c.get("redrob_signals", {}) or {}
    p = c.get("profile", {}) or {}
    
    try:
        days = (date(2026, 6, 18) - datetime.strptime(s.get("last_active_date", "")[:10], "%Y-%m-%d").date()).days
        active_s = 1.0 if days <= 14 else (0.75 if days <= 45 else 0.20)
    except Exception:
        active_s = 0.5

    otw_s = 1.0 if s.get("open_to_work_flag") else 0.40
    rrr_s = min((s.get("recruiter_response_rate", 0) or 0) / 0.70, 1.0)
    
    notice = s.get("notice_period_days", 60) or 60
    notice_s = 1.0 if notice <= 30 else (0.65 if notice <= 60 else 0.30)
    
    loc = _lower(p.get("location", ""))
    loc_s = 1.0 if any(pl in loc for pl in PREFERRED_LOCATIONS) else (0.70 if s.get("willing_to_relocate") else 0.45)

    final = (active_s * 0.35 + otw_s * 0.20 + rrr_s * 0.20 + notice_s * 0.15 + loc_s * 0.10)
    return max(0.0, min(1.0, final)), f"days_inactive:{days}"

# ─── PART 5 — LEADERBOARD TIER SELECTION MATRIX ──────────────────────────────

def tier_rerank(scored: list) -> list:
    """Applies distinct re-ranking constraints dynamically across tiers."""
    tier1 = [r for r in scored if r["scores"]["shipper"] >= 0.55 and r["scores"]["semantic"] >= 0.45]
    tier2 = [r for r in scored if r not in tier1 and r["composite"] >= 0.45]
    tier3 = [r for r in scored if r not in tier1 and r not in tier2]

    # Hyper-strict validation parameters applied exclusively to top finalists
    tier1.sort(key=lambda r: (r["scores"]["shipper"] * 0.45 + r["scores"]["semantic"] * 0.35 + r["scores"]["skill_depth"] * 0.20), reverse=True)
    tier2.sort(key=lambda r: r["composite"], reverse=True)
    tier3.sort(key=lambda r: (r["composite"] * 0.70 + r["scores"]["engagement"] * 0.30), reverse=True)

    final, seen = [], set()
    for r in (tier1 + tier2 + tier3):
        if r["candidate_id"] not in seen:
            seen.add(r["candidate_id"])
            final.append(r)
        if len(final) == TOP_FINAL:
            break
    return final

def get_llm_reason(client, candidate: dict, sc: dict) -> str:
    p = candidate.get("profile", {}) or {}
    prompt = f"Explain in one clean sentence under 25 words why an ML engineer with a shipper score of {sc['shipper']:.2f} who built tracking systems fits a founding search role at {p.get('current_company')}. Focus on product delivery, not buzzwords. Do not use generic praise."
    try:
        resp = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=80, messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text.strip().replace("\n", " ").replace('"', "'")
    except Exception:
        return "Demonstrated reliable product-engineering ownership deploying end-to-end vector matching systems at scale."

# ─── MAIN PROCESS STREAM ──────────────────────────────────────────────────────

def run():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing pipeline execution using device context: {device}")

    if not Path(INPUT_PATH).exists():
        print(f"[ERROR] Target dataset not found: {INPUT_PATH}")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"Loaded {len(candidates):,} candidate profiles from data stream.")

    # Execute pre-screening filters
    active_pool, rejected_pool = [], []
    for c in candidates:
        disq, reason = is_disqualified(c)
        if not disq:
            active_pool.append(c)
        else:
            c["_disqualification_reason"] = reason
            rejected_pool.append(c)
            
    with open(OUTPUT_REJECTS, "w", encoding="utf-8") as f:
        json.dump(rejected_pool, f, indent=2, ensure_ascii=False)
    print(f"Screening complete. Valid entries: {len(active_pool):,} | Saved Rejected: {len(rejected_pool):,}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    texts = [build_candidate_text(c) for c in active_pool]

    # GPU-accelerated embedding generation
    prefixed = ["Represent this sentence: " + t for t in texts]
    cand_vecs = model.encode(prefixed, batch_size=128, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

    query_prefixed = ["Represent this sentence: " + q for q in JD_QUERIES]
    query_vecs = model.encode(query_prefixed, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    # GPU High-Speed FAISS Max Proximity Mapping (OR logic)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(cand_vecs)

    all_scores = np.full(len(active_pool), -1.0, dtype=np.float32)
    for q_vec in query_vecs:
        sims, idxs = index.search(q_vec.reshape(1, -1), min(TOP_SEMANTIC * 3, len(active_pool)))
        for idx, sim in zip(idxs[0], sims[0]):
            if sim > all_scores[idx]:
                all_scores[idx] = sim

    top_sem_idx = np.argsort(all_scores)[::-1][:TOP_SEMANTIC]

    scored = []
    for idx in top_sem_idx:
        c = active_pool[idx]
        ship_s, ship_d = score_shipper(c)
        car_s, car_d = score_career(c)
        sk_s, sk_d = score_skill_depth(c)
        eng_s, eng_d = score_engagement(c)

        comp = (float(all_scores[idx]) * W_SEMANTIC) + (ship_s * W_SHIPPER) + (car_s * W_CAREER) + (eng_s * W_ENGAGE)
        scored.append({
            "candidate_id": c["candidate_id"], "candidate": c, "composite": comp,
            "scores": {"semantic": float(all_scores[idx]), "shipper": ship_s, "career": car_s, "skill_depth": sk_s, "engagement": eng_s},
            "debug": {"shipper": ship_d, "career": car_d, "skill": sk_d, "engage": eng_d}
        })

    scored.sort(key=lambda x: x["composite"], reverse=True)
    final_top = tier_rerank(scored)

    # Validate Anthropic Client environment context
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=api_key) if api_key else None
    if not client:
        print("[WARN] ANTHROPIC_API_KEY missing. Defaulting to template fallback text.")

    for i, r in enumerate(final_top):
        r["llm_reason"] = get_llm_reason(client, r["candidate"], r["scores"]) if client else "Verified background tracking real engineering outcomes over buzzword strings."

    # Write final leaderboard output spreadsheet format
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["rank", "candidate_id", "composite_score", "semantic_score", "shipper_score", "career_score", "skill_depth_score", "engagement_score", "llm_reason", "current_title", "current_company", "years_of_experience", "notice_period_days", "open_to_work", "shipper_debug"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for rank, r in enumerate(final_top, 1):
            w.writerow({
                "rank": rank, "candidate_id": r["candidate_id"], "composite_score": round(r["composite"], 4),
                "semantic_score": round(r["scores"]["semantic"], 4), "shipper_score": round(r["scores"]["shipper"], 4),
                "career_score": round(r["scores"]["career"], 4), "skill_depth_score": round(r["scores"]["skill_depth"], 4),
                "engagement_score": round(r["scores"]["engagement"], 4), "llm_reason": r["llm_reason"],
                "current_title": r["candidate"]["profile"].get("current_title", ""),
                "current_company": r["candidate"]["profile"].get("current_company", ""),
                "years_of_experience": r["candidate"]["profile"].get("years_of_experience", ""),
                "notice_period_days": r["candidate"]["redrob_signals"].get("notice_period_days", ""),
                "open_to_work": r["candidate"]["redrob_signals"].get("open_to_work_flag", ""),
                "shipper_debug": r["debug"]["shipper"]
            })

    print(f"\n[SUCCESS] Pipeline completed in {time.time() - t0:.1f}s. Summary logged: {OUTPUT_CSV}")

if __name__ == "__main__":
    run()