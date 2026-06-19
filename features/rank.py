# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — SCORE & EXPORT  (the submission run — 5 min, CPU only)        ║
# ║                                                                          ║
# ║  This script does ONLY cheap, pure-Python/NumPy work:                  ║
# ║    1. Loads top300_with_reasons.json (already has semantic_score        ║
# ║       and llm_reason precomputed by step1_precompute.py)                ║
# ║    2. Computes 4 lightweight scores from redrob_signals + career data:  ║
# ║       shipper, career, skill_depth, engagement                          ║
# ║    3. Combines into a composite score                                    ║
# ║    4. Applies tiered re-ranking (different strictness per rank band)    ║
# ║    5. Deduplicates and writes top100_final.csv                          ║
# ║                                                                          ║
# ║  NO model loading. NO embedding. NO API calls. NO FAISS.                ║
# ║  This is the script you run under the hackathon's 5-min/CPU constraint. ║
# ║                                                                          ║
# ║  HOW TO RUN:                                                            ║
# ║    !python step2_score_and_export.py                                     ║
# ║  (or just run this cell directly in Colab — no GPU needed)              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import json, csv, time
from pathlib import Path
from datetime import datetime, date

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

INPUT_PATH  = "/content/drive/MyDrive/redrob/outputs/top300_with_reasons.json"
OUTPUT_DIR  = "/content/drive/MyDrive/redrob/outputs"
TOP_FINAL   = 100

# Score weights (must sum to 1.0)
W_SEMANTIC = 0.35   # precomputed in step 1 — what they built matches the JD
W_SHIPPER  = 0.30   # evidence of shipping real systems (JD: tilt toward shipper)
W_CAREER   = 0.20   # right seniority, right company type
W_ENGAGE   = 0.15   # availability and behavioral signals


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "ltimindtree", "lti", "coforge", "birlasoft",
}

SHIPPING_EVIDENCE = [
    "deployed to production", "shipped to production", "in production",
    "serving real users", "deployed to real users", "live in production",
    "production traffic", "production system", "production environment",
    "at scale", "million users", "billion requests",
    "reduced latency", "p99", "p95", "throughput", "qps", "rps",
    "end-to-end", "owned the system", "led the development",
    "a/b test", "a/b testing", "online evaluation", "offline evaluation",
    "ndcg", "mrr", "precision@", "recall@", "evaluation framework",
    "ranking system", "retrieval system", "search system",
    "recommendation system", "recommender", "recsys",
    "hybrid retrieval", "dense retrieval", "bm25",
    "reranking", "re-ranking", "learning to rank",
    "semantic search", "vector search", "embedding-based",
    "candidate ranking", "job matching", "feedback loop",
]

RESEARCHER_SIGNALS = [
    "research scientist", "research intern", "postdoc", "post-doc",
    "phd student", "doctoral candidate", "thesis",
    "published paper", "arxiv", "research lab", "university lab",
    "ablation study", "sota", "novel approach", "dataset collection",
]

DOMAIN_SKILLS = {
    "elasticsearch", "opensearch", "solr", "faiss", "qdrant", "pinecone",
    "weaviate", "milvus", "bm25", "hnsw", "annoy", "vector search",
    "semantic search", "hybrid search", "information retrieval",
    "ranking", "learning to rank", "ltr", "lambdamart",
    "recommendation system", "recsys", "collaborative filtering",
    "xgboost", "lightgbm", "sentence-transformers", "bert", "transformers",
    "hugging face", "nlp", "natural language processing",
    "mlops", "model serving", "a/b testing", "ndcg", "mrr",
    "python", "pytorch", "tensorflow", "scikit-learn",
}

BUZZWORD_SKILLS = {
    "langchain", "openai", "chatgpt", "gpt-4", "gpt-3", "llama", "mistral",
    "rag", "generative ai", "stable diffusion", "llama index", "crewai",
    "autogpt", "langsmith", "copilot",
}

PREFERRED_LOCATIONS = {
    "noida", "pune", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "delhi", "gurugram", "gurgaon", "ncr", "chennai",
}


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _lower(t) -> str:
    return t.lower().strip() if t else ""

def _is_consulting(company: str) -> bool:
    return any(f in _lower(company) for f in CONSULTING_FIRMS)

def _days_ago(date_str: str) -> float:
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (date.today() - d).days
    except Exception:
        return 9999.0


# ═══════════════════════════════════════════════════════════════
#  SCORING FUNCTIONS — pure Python, no model needed
# ═══════════════════════════════════════════════════════════════

def score_shipper(c: dict) -> tuple[float, str]:
    """Evidence of shipping real production systems. Core JD requirement."""
    history = c.get("career_history", [])
    edu     = c.get("education", [])
    all_desc = " ".join(j.get("description", "") for j in history).lower()

    hits     = [p for p in SHIPPING_EVIDENCE if p in all_desc]
    ship_raw = min(len(hits), 10) / 10.0

    res_count = sum(
        1 for j in history
        if any(sig in _lower(j.get("title","") + " " + j.get("description",""))
               for sig in RESEARCHER_SIGNALS)
    )
    for e in edu:
        deg = _lower(e.get("degree", ""))
        if ("ph.d" in deg or "phd" in deg) and "production" not in all_desc and "deployed" not in all_desc:
            res_count += 2
    researcher_penalty = min(res_count * 0.07, 0.35)

    product_jobs  = sum(1 for j in history if not _is_consulting(j.get("company","")))
    product_ratio = product_jobs / max(len(history), 1)
    product_bonus = product_ratio * 0.15

    final = max(0.0, min(1.0, ship_raw * (1 - researcher_penalty) + product_bonus))
    note  = f"phrases:{len(hits)},res_penalty:{researcher_penalty:.2f},product_ratio:{product_ratio:.2f}"
    return final, note


def score_career(c: dict) -> tuple[float, str]:
    """Right seniority band, right domain, right company type."""
    p       = c.get("profile", {})
    history = c.get("career_history", [])

    yoe = float(p.get("years_of_experience", 0) or 0)
    if   5 <= yoe <= 9:   yoe_s = 1.0
    elif 4 <= yoe < 5:    yoe_s = 0.85
    elif 9 < yoe <= 12:   yoe_s = 0.80
    elif 3 <= yoe < 4:    yoe_s = 0.65
    elif 12 < yoe <= 15:  yoe_s = 0.65
    else:                 yoe_s = 0.40

    ML_TITLES = {
        "ml engineer", "machine learning", "ai engineer", "data scientist",
        "nlp engineer", "search engineer", "ranking engineer", "applied scientist",
        "backend engineer", "software engineer", "platform engineer",
        "senior engineer", "staff engineer", "principal engineer",
    }
    domain   = sum(1 for j in history if any(mt in _lower(j.get("title","")) for mt in ML_TITLES))
    domain_s = min(domain / max(len(history), 1), 1.0)

    avg_tenure = (sum(j.get("duration_months",0) or 0 for j in history) / len(history)
                  if history else 0)
    tenure_s = (1.0 if avg_tenure >= 24 else 0.80 if avg_tenure >= 18
                else 0.65 if avg_tenure >= 12 else 0.45)

    product_s = sum(1 for j in history if not _is_consulting(j.get("company",""))) / max(len(history), 1)

    final = yoe_s*0.30 + domain_s*0.35 + tenure_s*0.15 + product_s*0.20
    note  = f"yoe:{yoe}({yoe_s:.2f}),domain:{domain_s:.2f},tenure:{avg_tenure:.0f}mo,product:{product_s:.2f}"
    return max(0.0, min(1.0, final)), note


def score_skill_depth(c: dict) -> tuple[float, str]:
    """Depth over breadth. Penalises buzzword-only profiles (the keyword trap)."""
    skills        = c.get("skills", [])
    domain_depth  = 0.0
    buzzword_trap = 0

    for s in skills:
        name = _lower(s.get("name", ""))
        dur  = s.get("duration_months", 0) or 0
        end  = s.get("endorsements", 0) or 0
        is_dom = any(kw in name for kw in DOMAIN_SKILLS)
        is_buz = any(bw in name for bw in BUZZWORD_SKILLS)

        if is_buz and dur < 6 and end < 3:
            buzzword_trap += 1
            continue
        if is_dom:
            depth = (min(dur, 36) / 36) * 0.70 + (min(end, 20) / 20) * 0.30
            domain_depth += depth

    depth_norm   = min(domain_depth / 5.0, 1.0)
    trap_penalty = min((buzzword_trap / max(len(skills), 1)) * 0.40, 0.30)
    final = max(0.0, min(1.0, depth_norm - trap_penalty))
    note  = f"domain_depth:{domain_depth:.2f},buzzwords:{buzzword_trap},penalty:{trap_penalty:.2f}"
    return final, note


def score_engagement(c: dict) -> tuple[float, str]:
    """Availability and behavioral signals. Kept at 15% weight intentionally."""
    s = c.get("redrob_signals", {})
    p = c.get("profile", {})

    days_inactive = _days_ago(s.get("last_active_date", ""))
    active_s = (1.0 if days_inactive <= 7  else 0.85 if days_inactive <= 30
                else 0.65 if days_inactive <= 60 else 0.45 if days_inactive <= 90
                else 0.25 if days_inactive <= 180 else 0.10)

    otw_s = 1.0 if s.get("open_to_work_flag") else 0.40   # not 0.0 — soft signal only

    rrr_s = min((s.get("recruiter_response_rate", 0) or 0) / 0.70, 1.0)

    gh = s.get("github_activity_score", -1)
    gh_s = (1.0 if gh >= 70 else 0.75 if gh >= 40 else 0.50 if gh >= 20
            else 0.25 if gh >= 0 else 0.30)

    notice   = s.get("notice_period_days", 60) or 60
    notice_s = (1.0 if notice <= 15 else 0.85 if notice <= 30 else 0.65 if notice <= 60
                else 0.45 if notice <= 90 else 0.25)   # soft preference, NOT a hard filter

    loc   = _lower(p.get("location", ""))
    reloc = s.get("willing_to_relocate", False)
    loc_s = (1.0 if any(pl in loc for pl in PREFERRED_LOCATIONS)
             else 0.75 if reloc else 0.55)

    final = (active_s*0.30 + otw_s*0.20 + rrr_s*0.20 +
             gh_s*0.15 + notice_s*0.10 + loc_s*0.05)
    note  = (f"inactive:{days_inactive:.0f}d,otw:{s.get('open_to_work_flag')},"
             f"rrr:{s.get('recruiter_response_rate',0):.2f},gh:{gh},notice:{notice}d")
    return max(0.0, min(1.0, final)), note


def composite(sem, ship, career, engage) -> float:
    return sem*W_SEMANTIC + ship*W_SHIPPER + career*W_CAREER + engage*W_ENGAGE


# ═══════════════════════════════════════════════════════════════
#  TIERED RE-RANKING
#  Top 1-10:   Strictest — high shipper AND semantic
#  Top 11-50:  Balanced — high composite
#  Top 51-100: Broader — composite + engagement
# ═══════════════════════════════════════════════════════════════

def tier_rerank(scored: list) -> list:
    tier1 = [r for r in scored
             if r["scores"]["shipper"] >= 0.55 and r["scores"]["semantic"] >= 0.45]
    tier2 = [r for r in scored if r not in tier1 and r["composite"] >= 0.45]
    tier3 = [r for r in scored if r not in tier1 and r not in tier2]

    tier1.sort(key=lambda r: (
        r["scores"]["shipper"]*0.40 + r["scores"]["semantic"]*0.40 +
        r["scores"]["skill_depth"]*0.20
    ), reverse=True)
    tier2.sort(key=lambda r: r["composite"], reverse=True)
    tier3.sort(key=lambda r: r["composite"]*0.70 + r["scores"]["engagement"]*0.30,
               reverse=True)

    final, seen = [], set()
    for r in tier1 + tier2 + tier3:
        cid = r["candidate_id"]
        if cid not in seen:
            seen.add(cid)
            final.append(r)
        if len(final) == TOP_FINAL:
            break
    return final


# ═══════════════════════════════════════════════════════════════
#  CSV WRITER
# ═══════════════════════════════════════════════════════════════

def write_top100_csv(ranked: list, path: str):
    fieldnames = [
        "rank", "candidate_id",
        "composite_score", "semantic_score", "shipper_score",
        "career_score", "skill_depth_score", "engagement_score",
        "llm_reason",
        "current_title", "current_company", "years_of_experience", "location",
        "notice_period_days", "open_to_work", "github_activity_score",
        "shipper_debug", "career_debug", "skill_debug", "engage_debug",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for rank, r in enumerate(ranked, 1):
            c  = r["candidate"]
            p  = c.get("profile", {})
            s  = c.get("redrob_signals", {})
            sc = r["scores"]
            w.writerow({
                "rank"                : rank,
                "candidate_id"        : r["candidate_id"],
                "composite_score"     : round(r["composite"], 4),
                "semantic_score"      : round(sc["semantic"], 4),
                "shipper_score"       : round(sc["shipper"], 4),
                "career_score"        : round(sc["career"], 4),
                "skill_depth_score"   : round(sc["skill_depth"], 4),
                "engagement_score"    : round(sc["engagement"], 4),
                "llm_reason"          : r.get("llm_reason", ""),
                "current_title"       : p.get("current_title", ""),
                "current_company"     : p.get("current_company", ""),
                "years_of_experience" : p.get("years_of_experience", ""),
                "location"            : p.get("location", ""),
                "notice_period_days"  : s.get("notice_period_days", ""),
                "open_to_work"        : s.get("open_to_work_flag", ""),
                "github_activity_score": s.get("github_activity_score", ""),
                "shipper_debug"       : r["debug"]["shipper"],
                "career_debug"        : r["debug"]["career"],
                "skill_debug"         : r["debug"]["skill"],
                "engage_debug"        : r["debug"]["engage"],
            })


# ═══════════════════════════════════════════════════════════════
#  MAIN — fast, CPU-only, no models
# ═══════════════════════════════════════════════════════════════

def run():
    t0 = time.time()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        top300 = json.load(f)
    print(f"  Loaded {len(top300):,} pre-ranked candidates.")
    print(f"  (semantic_score and llm_reason were precomputed in step1)\n")

    print("  Scoring on redrob_signals + career data (pure Python, no model)...")
    scored = []
    for c in top300:
        sem_score = c.get("_semantic_score", 0.0)
        llm_reason = c.get("_llm_reason", "")

        ship_s,   ship_d   = score_shipper(c)
        career_s, career_d = score_career(c)
        skill_s,  skill_d  = score_skill_depth(c)
        engage_s, engage_d = score_engagement(c)
        comp = composite(sem_score, ship_s, career_s, engage_s)

        scored.append({
            "candidate_id": c["candidate_id"],
            "candidate"   : c,
            "composite"   : comp,
            "llm_reason"  : llm_reason,
            "scores": {
                "semantic"   : round(sem_score, 4),
                "shipper"    : round(ship_s, 4),
                "career"     : round(career_s, 4),
                "skill_depth": round(skill_s, 4),
                "engagement" : round(engage_s, 4),
            },
            "debug": {
                "shipper": ship_d, "career": career_d,
                "skill":   skill_d, "engage": engage_d,
            },
        })

    scored.sort(key=lambda r: r["composite"], reverse=True)
    print(f"  Scored {len(scored):,} candidates.\n")

    print("  Applying tiered re-ranking...")
    reranked = tier_rerank(scored)
    tier1_count = sum(1 for r in reranked
                      if r["scores"]["shipper"] >= 0.55 and r["scores"]["semantic"] >= 0.45)
    print(f"    Tier 1 (strict shipper+semantic) : {tier1_count}")
    print(f"    Tier 2+3 (balanced/broader)      : {len(reranked)-tier1_count}\n")

    seen_ids, deduped = set(), []
    for r in reranked:
        if r["candidate_id"] not in seen_ids:
            seen_ids.add(r["candidate_id"])
            deduped.append(r)
    final_top = deduped[:TOP_FINAL]
    if len(deduped) < len(reranked):
        print(f"  Removed {len(reranked)-len(deduped)} duplicate(s).\n")

    csv_path = out / "top100_final.csv"
    write_top100_csv(final_top, str(csv_path))

    elapsed = time.time() - t0
    print(f"""
╔══════════════════════════════════════════════════════════════╗
  SCORE & EXPORT COMPLETE  ({elapsed:.1f}s — well under 5 min)
╠══════════════════════════════════════════════════════════════╣
  Input (top semantic pool) : {len(top300):,}
  Final output              : {len(final_top)} (no duplicates)
  ────────────────────────────────────────────────────────────
  TOP 10:""")
    for i, r in enumerate(final_top[:10], 1):
        p  = r["candidate"]["profile"]
        sc = r["scores"]
        print(f"  #{i:>2}  {r['composite']:.3f}  "
              f"{p.get('current_title','')} @ {p.get('current_company','')}  "
              f"({p.get('years_of_experience',0)}y)")
        print(f"       sem={sc['semantic']:.2f} ship={sc['shipper']:.2f} "
              f"career={sc['career']:.2f} engage={sc['engagement']:.2f}")
        if r.get("llm_reason") and "skipped" not in r["llm_reason"]:
            print(f"       → {r['llm_reason']}")
    print(f"""  ────────────────────────────────────────────────────────────
  Output: {csv_path}      ← SUBMISSION FILE
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    run()