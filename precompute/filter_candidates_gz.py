"""
filter_candidates_gz.py
========================
Reads candidates.jsonl.gz and splits into two output files:
  - kept_candidates.json     → candidates who PASSED all filters
  - removed_candidates.json  → candidates who were DISQUALIFIED (with reason)

HOW TO USE IN GOOGLE COLAB:
────────────────────────────
Step 1 — Mount Drive (if your file is there):
    from google.colab import drive
    drive.mount('/content/drive')

Step 2 — Update the paths below and run:
    !python filter_candidates_gz.py

BUGS FIXED vs original script:
────────────────────────────────────────────────────────────────────────────────
BUG 1 — Non-tech keywords matched job *descriptions*, not just *titles*.
         e.g. "Optimized backend logistics data pipelines" was flagged as
         "logistics" even though the person was a Backend Engineer.
         FIX: _is_non_tech_role() is now only called on the job TITLE, never
              on the description text.

BUG 2 — The 80% non-tech career rule removed people who are currently in tech.
         e.g. someone who spent 5 years as an Ops Manager but is now an ML
         Engineer for 3 years would get removed unfairly.
         FIX: Removed the 80% rule entirely. Now we only disqualify when:
              • the candidate's CURRENT role is non-tech, AND
              • their CURRENT stint duration is ≥ 18 months.
              Anyone who has "left" a non-tech career and is now in tech: kept.

BUG 3 — LangChain check counted ALL historical LangChain months regardless of
         when they occurred, and the has_pre_llm safety net fired too late
         (after disqualification was already triggered).
         FIX: Now only counts LangChain/OpenAI months from RECENT jobs
              (is_current=True, or end_date within the last 24 months).
              The pre-LLM background check scans the ENTIRE career history,
              so someone with old BERT/XGBoost experience is protected even
              if they recently did some LangChain work.
────────────────────────────────────────────────────────────────────────────────
"""

import gzip
import json
import sys
import datetime
from pathlib import Path

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────
INPUT_PATH          = "../data/candidates.jsonl.gz"     # ← your .gz file path
KEPT_OUTPUT_PATH    = "../outputs/kept_candidates.json"    # ← candidates who passed
REMOVED_OUTPUT_PATH = "../outputs/removed_candidates.json" # ← candidates who failed
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies", "tech mahindra",
    "mphasis", "hexaware", "niit technologies", "mindtree", "l&t infotech",
    "ltimindtree", "lti", "coforge", "birlasoft", "mastek",
}

# These keywords are matched ONLY against job TITLES (never descriptions).
# That is the fix for Bug 1 — description text like "backend logistics pipelines"
# was being caught by "logistics" before.
NON_TECH_ROLE_KEYWORDS = {
    "marketing manager", "marketing executive", "marketing analyst",
    "content writer", "content creator", "copywriter", "brand manager",
    "social media manager", "seo specialist", "digital marketing",
    "hr manager", "human resources", "hr executive", "talent acquisition",
    "recruiter", "recruitment",
    "accountant", "accounting", "finance manager", "financial analyst",
    "bookkeeper", "auditor", "chartered accountant",
    "operations manager", "operations executive",
    "logistics manager", "supply chain manager",
    "procurement manager", "purchasing manager",
    "business development", "sales manager", "sales executive",
    "account manager", "customer success",
    "project manager", "program manager",
    "customer support", "customer service",
    "graphic designer",
    "civil engineer", "mechanical engineer",
    "administrative", "office manager",
}

RESEARCH_ONLY_SIGNALS = {
    "phd", "ph.d", "postdoc", "post-doc", "post doc",
    "research scientist", "research engineer",
    "academic lab", "university lab", "research lab",
    "professor", "assistant professor", "associate professor",
    "research fellow", "research associate",
}

PRODUCTION_SIGNALS = {
    "production", "deployed", "serving", "api", "pipeline",
    "real-time", "realtime", "live", "latency", "throughput",
    "inference", "serving infra", "mlops", "model serving",
    "a/b test", "a/b testing", "rollout",
}

LANGCHAIN_SIGNALS = {
    "langchain", "lang chain", "langsmith", "langgraph",
}

OPENAI_SIGNALS = {
    "openai", "open ai", "gpt-3", "gpt-4", "gpt3", "gpt4",
    "chatgpt", "chat gpt", "dall-e", "dalle",
}

# These are checked across the ENTIRE career (not just recent jobs).
# That is the fix for Bug 3 — someone who used BERT in 2019 should not get
# removed just because they used LangChain in a recent project.
PRE_LLM_ML_SKILLS = {
    "retrieval", "ranking", "bm25", "tfidf", "tf-idf", "elasticsearch",
    "lucene", "solr", "information retrieval", "learning to rank",
    "recsys", "recommendation system", "collaborative filtering",
    "xgboost", "lightgbm", "random forest", "gradient boosting",
    "sklearn", "scikit-learn", "spark ml", "mllib",
    "faiss", "annoy", "hnsw", "vector search",
    "nlp", "bert", "transformers", "word2vec", "glove",
}

# How far back to look for "recent" LangChain work (Bug 3 fix)
RECENT_MONTHS_WINDOW = 24


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _lower(text: str) -> str:
    return text.lower() if text else ""


def _contains_any(text: str, keywords: set) -> bool:
    t = _lower(text)
    return any(kw in t for kw in keywords)


def _is_consulting(company: str) -> bool:
    c = _lower(company)
    return any(firm in c for firm in CONSULTING_FIRMS)


def _is_non_tech_role(title: str) -> bool:
    """
    Check if a job TITLE is non-technical.
    IMPORTANT: Only pass job titles here — never descriptions.
    (Bug 1 fix: description text like 'backend logistics pipelines' must
    never be evaluated by this function.)
    """
    return _contains_any(title, NON_TECH_ROLE_KEYWORDS)


def _is_recent_job(job: dict) -> bool:
    """
    Returns True if the job is current or ended within the last 24 months.
    Used for the LangChain recency check (Bug 3 fix).
    """
    if job.get("is_current", False):
        return True
    end_date_str = job.get("end_date")
    if not end_date_str:
        return False
    try:
        end_date = datetime.datetime.strptime(end_date_str[:10], "%Y-%m-%d")
        cutoff = datetime.datetime.now() - datetime.timedelta(days=RECENT_MONTHS_WINDOW * 30)
        return end_date >= cutoff
    except (ValueError, TypeError):
        return False


def _candidate_text(candidate: dict) -> str:
    """Flatten all text fields into one searchable string (for broad skill checks)."""
    parts = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("company", ""))
        parts.append(job.get("description", ""))
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", ""))
    return " ".join(parts)


def _progress(current: int, total: int, kept: int, removed: int):
    if total == 0:
        return
    pct  = current / total * 100
    done = int(pct / 2)
    bar  = "█" * done + "░" * (50 - done)
    sys.stdout.write(
        f"\r  [{bar}] {pct:5.1f}%  |  "
        f"processed: {current:,}  kept: {kept:,}  removed: {removed:,}"
    )
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════
#  FILTER CHECKS
# ═══════════════════════════════════════════════════════════════

def check_consulting_only(candidate: dict):
    """
    Disqualify ONLY if every single job in the candidate's history is at a
    consulting firm.  If they have even one product-company job: keep.
    """
    history = candidate.get("career_history", [])
    if not history:
        return None

    all_consulting = all(_is_consulting(j.get("company", "")) for j in history)
    if not all_consulting:
        return None  # at least one non-consulting job → keep

    return (
        "consulting_only: entire career at consulting firms "
        "(TCS/Infosys/Wipro/etc.) with no product-company experience"
    )


def check_non_tech_role(candidate: dict):
    """
    BUG 2 FIX — Disqualify ONLY when:
      • The candidate's CURRENT role is non-technical, AND
      • Their current stint has lasted ≥ 18 months.

    This correctly handles:
      - Person who was Ops Manager for 5 yrs but is now ML Engineer → KEEP
        (current role is tech, so this check doesn't fire)
      - Person who is currently a Marketing Manager for 6 months → KEEP
        (current non-tech duration < 18 months)
      - Person who is currently an Operations Manager for 3+ years → REMOVE

    The old 80%-of-career rule is intentionally removed because it was
    disqualifying people who had made a legitimate career transition to tech.
    """
    history = candidate.get("career_history", [])
    if not history:
        return None

    # Find current job(s) — prefer is_current flag, fallback to null end_date
    current_jobs = [j for j in history if j.get("is_current", False)]
    if not current_jobs:
        current_jobs = [j for j in history if j.get("end_date") is None]
    if not current_jobs:
        return None  # cannot determine current role; don't penalise

    # There should normally be exactly one current job; take the first
    current_job = current_jobs[0]
    current_title    = current_job.get("title", "")
    current_duration = current_job.get("duration_months", 0) or 0

    # BUG 1 FIX: pass only the TITLE to _is_non_tech_role, never description
    if _is_non_tech_role(current_title) and current_duration >= 18:
        return (
            f"non_tech_role: currently '{current_title}' for "
            f"{current_duration:.0f} months (≥18 months in a non-tech role)"
        )

    return None


def check_pure_research(candidate: dict):
    """
    Disqualify if ≥60% of jobs are research-only AND no production signals
    are found anywhere in the candidate's history.
    """
    history = candidate.get("career_history", [])
    if not history:
        return None

    research_jobs   = 0
    production_seen = False

    for job in history:
        # For research/production detection, checking the description is fine —
        # these are specific technical signals, not role-category keywords.
        text = _lower(job.get("title", "") + " " + job.get("description", ""))
        if _contains_any(text, RESEARCH_ONLY_SIGNALS):
            research_jobs += 1
        if _contains_any(text, PRODUCTION_SIGNALS):
            production_seen = True

    if (research_jobs / len(history)) >= 0.6 and not production_seen:
        return (
            "pure_research: career spent in research environments "
            "with no production deployment evidence"
        )

    return None


def check_langchain_only_ai(candidate: dict):
    """
    BUG 3 FIX — Disqualify if:
      • Uses LangChain AND OpenAI wrappers in their profile, AND
      • RECENT LangChain/OpenAI work (last 24 months) totals < 12 months, AND
      • No pre-LLM production ML skills anywhere in career history.

    The key fixes:
      1. We count only RECENT jobs (is_current or ended < 24 months ago) for
         the LangChain month count.  Old jobs are ignored for this counter.
      2. We check PRE_LLM_ML_SKILLS across the ENTIRE career, so someone with
         a 2018–2020 BERT/XGBoost background is protected even if they only
         recently started using LangChain.
      3. Ordering is correct: pre-LLM check happens BEFORE the disqualification
         decision, not after.
    """
    full_text = _candidate_text(candidate)

    uses_langchain = _contains_any(full_text, LANGCHAIN_SIGNALS)
    uses_openai    = _contains_any(full_text, OPENAI_SIGNALS)

    if not (uses_langchain and uses_openai):
        return None  # not the LangChain-wrapper pattern → keep

    # Check pre-LLM background across ENTIRE career first
    has_pre_llm = _contains_any(full_text, PRE_LLM_ML_SKILLS)
    if has_pre_llm:
        return None  # solid ML background → keep regardless of LangChain usage

    # Count RECENT LangChain/OpenAI months only
    recent_langchain_months = 0.0
    for job in candidate.get("career_history", []):
        if not _is_recent_job(job):
            continue  # skip old jobs for this counter
        text = _lower(job.get("description", "") + " " + job.get("title", ""))
        if _contains_any(text, LANGCHAIN_SIGNALS) or _contains_any(text, OPENAI_SIGNALS):
            recent_langchain_months += job.get("duration_months", 0) or 0

    if recent_langchain_months < 12:
        return (
            f"langchain_only_ai: only {recent_langchain_months:.0f} recent months of "
            f"LangChain/OpenAI work and no pre-LLM production ML background"
        )

    return None  # ≥12 months of recent LangChain work → borderline, keep


# ═══════════════════════════════════════════════════════════════
#  MAIN DISPATCH
# ═══════════════════════════════════════════════════════════════

def should_disqualify(candidate: dict):
    """
    Run all checks in priority order.
    Returns (True, reason_string) or (False, None).
    """
    for check in [
        check_consulting_only,
        check_non_tech_role,
        check_pure_research,
        check_langchain_only_ai,
    ]:
        reason = check(candidate)
        if reason:
            return True, reason
    return False, None


# ═══════════════════════════════════════════════════════════════
#  WRITE HELPERS
# ═══════════════════════════════════════════════════════════════

def write_json_array(filepath: Path, records: list):
    """Write a list of dicts as a pretty JSON array."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, record in enumerate(records):
            comma = "," if i < len(records) - 1 else ""
            f.write("  " + json.dumps(record, ensure_ascii=False) + comma + "\n")
        f.write("]\n")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_filter():
    input_path   = Path(INPUT_PATH)
    kept_path    = Path(KEPT_OUTPUT_PATH)
    removed_path = Path(REMOVED_OUTPUT_PATH)

    if not input_path.exists():
        print(f"\n[ERROR] File not found: {input_path}")
        print("  → Update INPUT_PATH at the top of this script.")
        return

    kept_list    = []
    removed_list = []
    total        = 0
    skipped      = 0

    print(f"\n  Reading: {input_path}")
    print(f"  This may take a minute for large files...\n")

    with gzip.open(input_path, "rt", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            disqualified, reason = should_disqualify(candidate)

            if disqualified:
                candidate_with_reason = dict(candidate)
                candidate_with_reason["filter_reason"] = reason
                removed_list.append(candidate_with_reason)
            else:
                kept_list.append(candidate)

            if total % 500 == 0:
                _progress(total, total, len(kept_list), len(removed_list))

    _progress(total, total, len(kept_list), len(removed_list))
    print()

    print(f"\n  Writing kept candidates → {kept_path} ...")
    write_json_array(kept_path, kept_list)

    print(f"  Writing removed candidates → {removed_path} ...")
    write_json_array(removed_path, removed_list)

    removal_breakdown = {}
    for c in removed_list:
        tag = c.get("filter_reason", "unknown").split(":")[0]
        removal_breakdown[tag] = removal_breakdown.get(tag, 0) + 1

    print(f"""
╔══════════════════════════════════════════════════════╗
  FILTER COMPLETE
╠══════════════════════════════════════════════════════╣
  Total processed  : {total:,}
  Malformed lines  : {skipped:,}
  ──────────────────────────────────────────────────
  ✅ Kept          : {len(kept_list):,}
  ❌ Removed       : {len(removed_list):,}
  ──────────────────────────────────────────────────
  Removal breakdown:""")
    for tag, count in sorted(removal_breakdown.items(), key=lambda x: -x[1]):
        print(f"    {tag:<25} {count:,}")
    print(f"""  ──────────────────────────────────────────────────
  Output files:
    {kept_path}
    {removed_path}
╚══════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    run_filter()