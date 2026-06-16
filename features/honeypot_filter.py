"""
Candidate Filtering Script
--------------------------
Reads a .gz file (one JSON candidate per line), applies two filters:

  1. CONSULTING-ONLY FILTER — Remove candidates whose ENTIRE career has been
     at consulting/body-shopping firms and who have never worked at a product
     or non-consulting company.

  2. NON-TECH ROLE FILTER — Remove candidates whose CURRENT role is a
     non-technical business role (BA, marketing, HR, operations, etc.) AND
     who have no meaningful tech engineering career history.

Outputs:
  - <input_stem>_filtered.json   (JSON array of remaining candidates)
  - <input_stem>_filtered.jsonl  (one JSON object per line of remaining candidates)
  - <input_stem>_removed.jsonl   (one JSON object per line of removed candidates)
"""

import gzip
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Firms considered "consulting / body-shopping" for the consulting-only check.
# A candidate is only filtered if ALL their jobs are at firms in this set.
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant", "cognizant technology solutions",
    "capgemini",
    "hcl", "hcl technologies",
    "tech mahindra",
    "mphasis",
    "mindtree",
    "hexaware",
    "l&t infotech", "l&t technology services", "ltimindtree",
    "niit technologies",
    "patni",
    "mastech",
    "kpit",
    "zensar",
    "birlasoft",
    "sonata software",
}

# Job titles that are considered NON-technical.
# Candidates whose entire career (or current title) falls into these buckets
# and who have no tech-engineering history will be removed.
NON_TECH_TITLES = {
    # Business / strategy
    "business analyst", "business development", "management consultant",
    "strategy consultant", "management trainee",
    # Marketing
    "marketing manager", "marketing executive", "digital marketing",
    "content writer", "content marketing", "seo specialist", "copywriter",
    "brand manager", "growth manager",
    # Sales
    "sales executive", "sales manager", "account executive",
    "account manager", "business development executive",
    # HR / People
    "hr manager", "human resources", "hr executive", "recruiter",
    "talent acquisition", "people operations",
    # Finance / Accounting
    "accountant", "finance manager", "financial analyst", "chartered accountant",
    "accounts executive",
    # Operations / Admin
    "operations manager", "operations executive", "customer support",
    "customer success", "customer service", "project manager",
    "program manager", "delivery manager", "civil engineer",
    "mechanical engineer", "graphic designer",
}

# Keywords that, if found in a job title, mark it as a TECH role.
TECH_TITLE_KEYWORDS = {
    "engineer", "developer", "architect", "programmer", "scientist",
    "analyst",          # data analyst, ml analyst  – but NOT business analyst (handled separately)
    "devops", "sre", "reliability",
    "data", "machine learning", "ml", "ai ", "nlp", "cv",
    "backend", "frontend", "full stack", "fullstack", "full-stack",
    "cloud", "infrastructure", "platform", "security",
    "qa", "quality assurance", "test automation",
    "mobile", "android", "ios",
    "blockchain", "embedded",
    "research",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lower-case and strip extra whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def company_is_consulting(company_name: str) -> bool:
    name = normalise(company_name)
    return any(cf in name for cf in CONSULTING_FIRMS)


def title_is_tech(title: str) -> bool:
    t = normalise(title)
    # Explicitly exclude "business analyst" even though it contains "analyst"
    if "business analyst" in t:
        return False
    return any(kw in t for kw in TECH_TITLE_KEYWORDS)


def title_is_non_tech(title: str) -> bool:
    t = normalise(title)
    return any(nt in t for nt in NON_TECH_TITLES)


def has_any_non_consulting_experience(career: list) -> bool:
    """
    Returns True if the candidate has at least one job at a company that is
    NOT in our consulting-firm list.
    """
    for job in career:
        company = job.get("company", "")
        if not company_is_consulting(company):
            return True
    return False


def has_tech_engineering_history(career: list) -> bool:
    """
    Returns True if the candidate has at least one job whose title looks
    like a tech/engineering role.
    """
    for job in career:
        title = job.get("title", "")
        if title_is_tech(title):
            return True
    return False


# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------

def should_keep(candidate: dict) -> tuple[bool, str]:
    """
    Returns (keep: bool, reason: str).
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    current_title = normalise(profile.get("current_title", ""))
    cid = candidate.get("candidate_id", "?")

    # ── FILTER 1: Consulting-only careers ──────────────────────────────────
    # Only remove if every single company they've worked at is a consulting firm.
    if career and not has_any_non_consulting_experience(career):
        return False, f"{cid}: pure consulting career"

    # ── FILTER 2: Non-tech current role with no engineering history ─────────
    # Remove if:
    #   a) current title is a non-tech role, AND
    #   b) the candidate has no tech-engineering title anywhere in their history
    if title_is_non_tech(current_title) and not has_tech_engineering_history(career):
        return False, f"{cid}: non-tech role '{current_title}' with no engineering history"

    return True, ""


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_candidates(gz_path: Path) -> list[dict]:
    candidates = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] Skipping malformed line: {exc}", file=sys.stderr)
    return candidates


def main():
    # Check if we are running in an interactive notebook or with dummy system flags
    if len(sys.argv) < 2 or sys.argv[1].startswith('-f'):
        # Fallback path pointing directly to your file from the sidebar
        gz_path = Path("candidates.jsonl.gz")
    else:
        gz_path = Path(sys.argv[1])

    if not gz_path.exists():
        print(f"Error: file not found — {gz_path}", file=sys.stderr)
        sys.exit(1)

    # Default output name: same stem, _filtered.json
    if len(sys.argv) >= 3 and not sys.argv[2].startswith('-f'):
        out_path = Path(sys.argv[2])
    else:
        stem = gz_path.name.replace(".jsonl.gz", "").replace(".gz", "")
        out_path = gz_path.parent / f"{stem}_filtered.json"

    # Derive additional output file paths based on requirements
    out_path_jsonl = out_path.with_suffix(".jsonl")
    removed_path_jsonl = out_path.parent / f"{out_path.stem.replace('_filtered', '')}_removed.jsonl"

    print(f"Loading candidates from: {gz_path}")
    candidates = load_candidates(gz_path)
    total_candidates = len(candidates)
    print(f"  Loaded {total_candidates} candidates")

    kept, removed = [], []
    removed_objects = []  # To store the full candidate objects that got filtered out

    for c in candidates:
        keep, reason = should_keep(c)
        if keep:
            kept.append(c)
        else:
            removed.append((c.get("candidate_id", "?"), reason))
            # Keep a copy of the candidate data and append the exact exclusion reason
            rc = c.copy()
            rc["exclusion_reason"] = reason
            removed_objects.append(rc)

    # Detailed itemized list of removals (Maintained exactly as original)
    print(f"\n  Removed {len(removed)} candidates:")
    for cid, reason in removed:
        print(f"    ✗  {reason}")

    # New descriptive dynamic summary statements 
    print("\n" + "="*50)
    print("  FILTERING METRICS SUMMARY")
    print("="*50)
    print(f"  Removed {len(removed)} candidates out of {total_candidates} total candidates.")
    
    # Simple percentage visualization for readability 
    if total_candidates > 0:
        pct_removed = (len(removed) / total_candidates) * 100
        pct_kept = (len(kept) / total_candidates) * 100
        print(f"  ↳ Filtered out: {pct_removed:.1f}% of the dataset.")
        print(f"  ↳ Retained:    {pct_kept:.1f}% of the dataset.")
    
    print(f"  Final remaining pool: {len(kept)} candidates.")
    print("="*50)

    # 1. Write the clean JSON array file
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, indent=2, ensure_ascii=False)
    print(f"\nOutput successfully written to (JSON Array): {out_path}")

    # 2. Write the retained candidates to a JSONL file
    with open(out_path_jsonl, "w", encoding="utf-8") as fh:
        for candidate in kept:
            fh.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    print(f"Output successfully written to (JSONL Retained): {out_path_jsonl}")

    # 3. Write the removed candidates to a separate JSONL file
    with open(removed_path_jsonl, "w", encoding="utf-8") as fh:
        for candidate in removed_objects:
            fh.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    print(f"Output successfully written to (JSONL Removed) : {removed_path_jsonl}")


if __name__ == "__main__":
    main()