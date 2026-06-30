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
# ║       with EXACTLY 4 columns: candidate_id, rank, score, reasoning      ║
# ║                                                                          ║
# ║  DETERMINISM FIX                                                        ║
# ║  ────────────────────────────────────────────────────                    ║
# ║  Every sort in this script uses a STABLE secondary key (candidate_id)   ║
# ║  so re-running on the same top300_with_reasons.json input always        ║
# ║  produces byte-identical output. Python's sort is stable, but ties on   ║
# ║  floating-point composite scores were previously unordered — fixed by   ║
# ║  always tie-breaking on candidate_id.                                   ║
# ║                                                                          ║
# ║  NO model loading. NO embedding. NO API calls. NO FAISS.                ║
# ║  This is the script you run under the hackathon's 5-min/CPU constraint. ║
# ║                                                                          ║
# ║  HOW TO RUN:                                                            ║
# ║    !python step2_score_and_export.py                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import json, csv, time
from pathlib import Path
from datetime import datetime, date

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

# INPUT_PATH  = "../outputs/top300_with_reasons.json"
# Replace line 21 in runtime/rank.py with this if it throws a FileNotFoundError inside the function:
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "outputs", "top300_with_reasons.json")
OUTPUT_DIR  = "../outputs"
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
    "million users", "billion requests",
    "reduced latency", "p99", "p95", "throughput", "qps", "rps",
    "a/b test", "a/b testing", "online evaluation", "offline evaluation",
    "ndcg", "mrr", "precision@", "recall@", "evaluation framework",
    "ranking system", "retrieval system", "search system",
    "recommendation system", "recommender system", "recsys",
    "hybrid retrieval", "dense retrieval", "bm25",
    "reranking", "re-ranking", "learning to rank",
    "semantic search", "vector search", "embedding-based retrieval",
    "candidate ranking system", "job matching system",
]
# NOTE: removed generic phrases ("end-to-end", "led the development", "at scale",
# "owned the system", "feedback loop") that previously leaked into non-technical
# descriptions (e.g. HR process improvements). Every remaining phrase is specific
# to ML/search/ranking system-building and unlikely to appear in HR/marketing text.

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

# Same technical-role gate as step 1 — defends against non-technical
# candidates reaching this stage if step1's input file was ever edited
# or merged with an un-gated source after the fact.
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
# the bare substring "engineer" would match "Civil Engineer" or "Mechanical
# Engineer" — caught during testing on real sample data where these slipped
# through the gate (e.g. CAND_0000045: Project Manager with a prior Civil
# Engineer role would otherwise pass).
NON_SOFTWARE_ENGINEER_TITLES = [
    "civil engineer", "mechanical engineer", "electrical engineer",
    "chemical engineer", "structural engineer", "industrial engineer",
    "petroleum engineer", "mining engineer", "aerospace engineer",
    "environmental engineer", "biomedical engineer", "agricultural engineer",
]


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

def is_technical_candidate(c: dict) -> bool:
    """
    Same gate as step1 — defends against non-technical candidates reaching
    this stage if step1's input file was ever edited or merged with an
    un-gated source after the fact.
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
    if c.get("redrob_signals", {}).get("skill_assessment_scores"):
        return True
    for job in c.get("career_history", []):
        if _is_tech_title(job.get("title") or ""):
            return True
    return False


# ═══════════════════════════════════════════════════════════════
#  SCORING FUNCTIONS — pure Python, no model needed
# ═══════════════════════════════════════════════════════════════

def score_shipper(c: dict) -> float:
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

    return max(0.0, min(1.0, ship_raw * (1 - researcher_penalty) + product_bonus))


def score_career(c: dict) -> float:
    p       = c.get("profile", {})
    history = c.get("career_history", [])

    # HARD GATE: non-technical candidates cap at 0.15 career score regardless
    # of tenure/seniority/product-company status. This is what was missing
    # before — domain mismatch must dominate, not just contribute 35% weight.
    if not is_technical_candidate(c):
        return 0.15

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
    return max(0.0, min(1.0, final))


def score_skill_depth(c: dict) -> float:
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
    return max(0.0, min(1.0, depth_norm - trap_penalty))


def score_engagement(c: dict) -> float:
    """
    Uses redrob_signals fields:
      last_active_date, open_to_work_flag, recruiter_response_rate,
      github_activity_score, notice_period_days, willing_to_relocate
    """
    s = c.get("redrob_signals", {})
    p = c.get("profile", {})

    days_inactive = _days_ago(s.get("last_active_date", ""))
    active_s = (1.0 if days_inactive <= 7  else 0.85 if days_inactive <= 30
                else 0.65 if days_inactive <= 60 else 0.45 if days_inactive <= 90
                else 0.25 if days_inactive <= 180 else 0.10)

    otw_s = 1.0 if s.get("open_to_work_flag") else 0.40

    rrr_s = min((s.get("recruiter_response_rate", 0) or 0) / 0.70, 1.0)

    gh = s.get("github_activity_score", -1)
    gh_s = (1.0 if gh >= 70 else 0.75 if gh >= 40 else 0.50 if gh >= 20
            else 0.25 if gh >= 0 else 0.30)

    notice   = s.get("notice_period_days", 60) or 60
    notice_s = (1.0 if notice <= 15 else 0.85 if notice <= 30 else 0.65 if notice <= 60
                else 0.45 if notice <= 90 else 0.25)

    loc   = _lower(p.get("location", ""))
    reloc = s.get("willing_to_relocate", False)
    loc_s = (1.0 if any(pl in loc for pl in PREFERRED_LOCATIONS)
             else 0.75 if reloc else 0.55)

    return max(0.0, min(1.0, (
        active_s*0.30 + otw_s*0.20 + rrr_s*0.20 +
        gh_s*0.15 + notice_s*0.10 + loc_s*0.05
    )))


def composite(sem, ship, career, engage) -> float:
    return sem*W_SEMANTIC + ship*W_SHIPPER + career*W_CAREER + engage*W_ENGAGE


# ═══════════════════════════════════════════════════════════════
#  TIERED RE-RANKING (deterministic — every sort tie-breaks on candidate_id)
# ═══════════════════════════════════════════════════════════════

def tier_rerank(scored: list) -> list:
    tier1 = [r for r in scored
             if r["shipper"] >= 0.55 and r["semantic"] >= 0.45]
    tier1_ids = {r["candidate_id"] for r in tier1}

    tier2 = [r for r in scored
             if r["candidate_id"] not in tier1_ids and r["composite"] >= 0.45]
    tier2_ids = {r["candidate_id"] for r in tier2}

    tier3 = [r for r in scored
             if r["candidate_id"] not in tier1_ids and r["candidate_id"] not in tier2_ids]

    # Every sort key includes candidate_id as a tie-breaker so identical
    # scores always resolve to the same order across repeated runs.
    tier1.sort(key=lambda r: (
        -(r["shipper"]*0.40 + r["semantic"]*0.40 + r["skill_depth"]*0.20),
        r["candidate_id"]
    ))
    tier2.sort(key=lambda r: (-r["composite"], r["candidate_id"]))
    tier3.sort(key=lambda r: (
        -(r["composite"]*0.70 + r["engagement"]*0.30),
        r["candidate_id"]
    ))

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
#  CSV WRITER — exactly 4 columns as requested
# ═══════════════════════════════════════════════════════════════

def write_top100_csv(ranked: list, path: str):
    fieldnames = ["candidate_id", "rank", "score", "reasoning"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, r in enumerate(ranked, 1):
            w.writerow({
                "candidate_id": r["candidate_id"],
                "rank"        : rank,
                "score"       : round(r["composite"], 4),
                "reasoning"   : r.get("llm_reason", ""),
            })


# ═══════════════════════════════════════════════════════════════
#  MAIN — fast, CPU-only, no models, fully deterministic
# ═══════════════════════════════════════════════════════════════

def run():
    t0 = time.time()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        top300 = json.load(f)
    print(f"  Loaded {len(top300):,} pre-ranked candidates.\n")

    print("  Scoring on redrob_signals + career data (pure Python, no model)...")
    scored = []
    for c in top300:
        sem_score = c.get("_semantic_score", 0.0)
        llm_reason = c.get("_llm_reason", "")

        ship_s   = score_shipper(c)
        career_s = score_career(c)
        skill_s  = score_skill_depth(c)
        engage_s = score_engagement(c)
        comp     = composite(sem_score, ship_s, career_s, engage_s)

        scored.append({
            "candidate_id": c["candidate_id"],
            "composite"   : comp,
            "semantic"    : round(sem_score, 4),
            "shipper"     : round(ship_s, 4),
            "career"      : round(career_s, 4),
            "skill_depth" : round(skill_s, 4),
            "engagement"  : round(engage_s, 4),
            "llm_reason"  : llm_reason,
        })

    # Deterministic primary sort: composite desc, tie-break candidate_id asc
    scored.sort(key=lambda r: (-r["composite"], r["candidate_id"]))
    print(f"  Scored {len(scored):,} candidates.\n")

    print("  Applying tiered re-ranking...")
    reranked = tier_rerank(scored)
    tier1_count = sum(1 for r in reranked if r["shipper"] >= 0.55 and r["semantic"] >= 0.45)
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
  CSV columns                : candidate_id, rank, score, reasoning
  Determinism                : sorts tie-break on candidate_id —
                                re-running on same input gives same file
  ────────────────────────────────────────────────────────────
  TOP 10:""")
    for i, r in enumerate(final_top[:10], 1):
        print(f"  #{i:>2}  {r['composite']:.3f}  {r['candidate_id']}")
        print(f"       sem={r['semantic']:.2f} ship={r['shipper']:.2f} "
              f"career={r['career']:.2f} engage={r['engagement']:.2f}")
        if r.get("llm_reason") and "skipped" not in r["llm_reason"]:
            print(f"       → {r['llm_reason']}")
    print(f"""  ────────────────────────────────────────────────────────────
  Output: {csv_path}      ← SUBMISSION FILE
╚══════════════════════════════════════════════════════════════╝
""")
    
    return final_top


if __name__ == "__main__":
    run()