"""
remove_twins.py  — One-pass behavioral twin detection
======================================================
Replaces the slow two-pass O(n²) approach with a fast
MinHash + LSH (Locality Sensitive Hashing) method.

WHY THE OLD CODE WAS WRONG:
────────────────────────────
1. Pass 1 (exact description fingerprint) treated shared JOB DESCRIPTION
   TEMPLATES as twins. In a synthetic dataset, 16+ candidates share the
   same "Customer support team lead" description — they are NOT the same
   person. Grouping them together removed thousands of real candidates.

2. Pass 2 (SequenceMatcher on all pairs) is O(n²). With 2000 candidates
   after Pass 1 that is already 2 million comparisons = 30+ minutes CPU.
   With 10k it becomes hours.

WHY THIS CODE IS CORRECT AND FAST:
────────────────────────────────────
MinHash + LSH is the industry standard for large-scale near-duplicate
detection. It works in O(n) time (not O(n²)) and finds candidates whose
COMPLETE PROFILE FINGERPRINT is nearly identical — not just candidates
who share one or two templated job description sentences.

A "behavioral twin" means:
  Same name + Same YOE + Same current title + Same career arc + Same skills
  → this is the same person who uploaded their profile twice with minor edits.

It does NOT mean:
  Two different people who happened to work in the same industry and got
  the same template description in a synthetic dataset.

WHAT WE USE TO BUILD THE FINGERPRINT (not descriptions alone):
  - Candidate name (normalized)
  - Years of experience (binned to nearest 0.5y)
  - Current title + company
  - All job titles in career (order-independent)
  - All company names in career (order-independent)
  - Top 8 skills by proficiency/endorsements

INSTALL (Google Colab):
  !pip install datasketch --quiet

HOW TO RUN:
  !python remove_twins.py
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    print("[ERROR] datasketch not installed.")
    print("  Run: !pip install datasketch --quiet")
    sys.exit(1)

# ── CONFIGURE ────────────────────────────────────────────────────────────────
INPUT_PATH         = "kept_candidates.json"      # ← your filtered file
UNIQUE_OUTPUT_PATH = "cleaned_candidates.json"   # ← twins removed
TWINS_OUTPUT_PATH  = "removed_twins.json"        # ← what was removed

# Similarity threshold: 0.85 = profiles must share 85% of their shingles.
# At this threshold, only near-identical profiles (same person, minor edits)
# will be flagged. Raising it to 0.90 = stricter (fewer removed).
SIMILARITY_THRESHOLD = 0.85

# MinHash parameters: more permutations = more accurate but slower.
# 128 is the standard sweet spot for this task.
NUM_PERMUTATIONS = 128
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════
#  FINGERPRINT BUILDER
#  We build a SET of "shingles" — meaningful tokens from the
#  candidate profile. Two candidates are twins if their shingle
#  sets overlap by ≥ SIMILARITY_THRESHOLD (Jaccard similarity).
# ═══════════════════════════════════════════════════════════════

def _norm(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def _bin_yoe(yoe) -> str:
    """Bin years of experience to nearest 0.5 to allow minor rounding differences."""
    try:
        return str(round(float(yoe) * 2) / 2)
    except Exception:
        return "unknown"

def build_shingles(candidate: dict) -> set[bytes]:
    """
    Build a set of meaningful tokens (shingles) from a candidate profile.

    We use STRUCTURAL signals — name, titles, companies, skills —
    NOT job description text. Job descriptions in synthetic datasets
    are templated and shared across many different real candidates,
    so using them as fingerprints causes massive false-positive twins.

    Each shingle is a namespaced token like b"skill:pytorch" or
    b"title:senior ml engineer" so tokens from different fields
    can't accidentally collide with each other.
    """
    shingles = set()
    profile  = candidate.get("profile", {})
    history  = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])

    # 1. Name (most reliable single signal)
    name = _norm(profile.get("anonymized_name", ""))
    if name:
        shingles.add(f"name:{name}".encode())

    # 2. Years of experience (binned)
    yoe = _bin_yoe(profile.get("years_of_experience"))
    shingles.add(f"yoe:{yoe}".encode())

    # 3. Current title (normalised)
    title = _norm(profile.get("current_title", ""))
    if title:
        shingles.add(f"cur_title:{title}".encode())

    # 4. Current company (normalised)
    company = _norm(profile.get("current_company", ""))
    if company:
        shingles.add(f"cur_company:{company}".encode())

    # 5. Every (title, company) pair in career history
    #    Order-independent: sorted so profile reordering doesn't matter
    for job in history:
        jt = _norm(job.get("title", ""))
        jc = _norm(job.get("company", ""))
        if jt:
            shingles.add(f"title:{jt}".encode())
        if jc:
            shingles.add(f"company:{jc}".encode())
        if jt and jc:
            shingles.add(f"job:{jt}@{jc}".encode())

    # 6. Top skills (up to 8 — sorted so order doesn't matter)
    sorted_skills = sorted(
        skills,
        key=lambda s: (s.get("endorsements", 0), s.get("duration_months", 0)),
        reverse=True
    )
    for skill in sorted_skills[:8]:
        sn = _norm(skill.get("name", ""))
        if sn:
            shingles.add(f"skill:{sn}".encode())

    # 7. Education: field of study (not institution — too noisy)
    for edu in candidate.get("education", [])[:1]:
        field = _norm(edu.get("field_of_study", ""))
        if field:
            shingles.add(f"edu_field:{field}".encode())

    return shingles


# ═══════════════════════════════════════════════════════════════
#  MINHASH BUILDER
# ═══════════════════════════════════════════════════════════════

def build_minhash(shingles: set[bytes]) -> MinHash:
    m = MinHash(num_perm=NUM_PERMUTATIONS)
    for s in shingles:
        m.update(s)
    return m


# ═══════════════════════════════════════════════════════════════
#  TWIN REMOVAL — SINGLE PASS
# ═══════════════════════════════════════════════════════════════

def _best_in_pair(c1: dict, c2: dict) -> tuple[dict, dict]:
    """
    Given two twins, decide which to keep and which to remove.
    Priority (in order):
      1. More complete profile (profile_completeness_score)
      2. More recently active (last_active_date)
      3. Higher recruiter response rate
    """
    s1 = c1["redrob_signals"]
    s2 = c2["redrob_signals"]

    score1 = s1.get("profile_completeness_score", 0)
    score2 = s2.get("profile_completeness_score", 0)
    if score1 != score2:
        return (c1, c2) if score1 > score2 else (c2, c1)

    active1 = s1.get("last_active_date", "0000")
    active2 = s2.get("last_active_date", "0000")
    if active1 != active2:
        return (c1, c2) if active1 > active2 else (c2, c1)

    rr1 = s1.get("recruiter_response_rate", 0)
    rr2 = s2.get("recruiter_response_rate", 0)
    return (c1, c2) if rr1 >= rr2 else (c2, c1)


def _progress(current: int, total: int, label: str = ""):
    pct  = current / total * 100 if total else 0
    done = int(pct / 2)
    bar  = "█" * done + "░" * (50 - done)
    print(f"\r  [{bar}] {pct:5.1f}%  {label}", end="", flush=True)


def remove_twins(candidates: list) -> tuple[list, list]:
    """
    Single-pass MinHash + LSH twin detection.

    Steps:
      1. Build a shingle set + MinHash for every candidate.
      2. Insert each MinHash into an LSH index.
      3. For each candidate, query the index for near-duplicates.
      4. When a twin pair is found, keep the better profile, remove the other.
      5. Never re-process an already-removed candidate.

    Time complexity: O(n × k) where k = NUM_PERMUTATIONS (constant).
    This is effectively O(n) — linear in the number of candidates.
    """
    n = len(candidates)
    print(f"\n  Building MinHash fingerprints for {n:,} candidates...")

    # Build a MinHash per candidate
    minhashes = {}
    for i, c in enumerate(candidates):
        cid      = c["candidate_id"]
        shingles = build_shingles(c)
        minhashes[cid] = build_minhash(shingles)
        if (i + 1) % 1000 == 0 or (i + 1) == n:
            _progress(i + 1, n, f"{i+1:,}/{n:,}")
    print()  # newline after bar

    # Build LSH index
    print(f"\n  Building LSH index (threshold={SIMILARITY_THRESHOLD})...")
    lsh = MinHashLSH(threshold=SIMILARITY_THRESHOLD, num_perm=NUM_PERMUTATIONS)
    for cid, mh in minhashes.items():
        lsh.insert(cid, mh)

    # Query phase — single pass
    print(f"  Querying index for twins...\n")
    removed_ids    = set()   # candidate_ids to remove
    twin_pairs     = []      # (keep_id, remove_id) pairs found
    candidate_map  = {c["candidate_id"]: c for c in candidates}

    for i, c in enumerate(candidates):
        cid = c["candidate_id"]

        if cid in removed_ids:
            continue

        # Query returns all candidates within threshold of this one
        results = lsh.query(minhashes[cid])
        # Remove self from results
        neighbors = [r for r in results if r != cid and r not in removed_ids]

        for neighbor_id in neighbors:
            c1 = candidate_map[cid]
            c2 = candidate_map[neighbor_id]
            keep, remove = _best_in_pair(c1, c2)

            removed_ids.add(remove["candidate_id"])
            twin_pairs.append((keep["candidate_id"], remove["candidate_id"]))

    # Split into kept and removed lists
    kept    = [c for c in candidates if c["candidate_id"] not in removed_ids]
    removed = []
    for c in candidates:
        if c["candidate_id"] in removed_ids:
            # Find what it was twinned with
            pair_info = next(
                (p for p in twin_pairs if p[1] == c["candidate_id"]), None
            )
            c_copy = dict(c)
            c_copy["_twin_of"] = pair_info[0] if pair_info else "unknown"
            removed.append(c_copy)

    return kept, removed


# ═══════════════════════════════════════════════════════════════
#  WRITE OUTPUT
# ═══════════════════════════════════════════════════════════════

def write_json_array(filepath: Path, records: list):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, record in enumerate(records):
            comma = "," if i < len(records) - 1 else ""
            f.write("  " + json.dumps(record, ensure_ascii=False) + comma + "\n")
        f.write("]\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def run():
    input_path   = Path(INPUT_PATH)
    unique_path  = Path(UNIQUE_OUTPUT_PATH)
    twins_path   = Path(TWINS_OUTPUT_PATH)

    if not input_path.exists():
        print(f"\n[ERROR] File not found: {input_path}")
        print("  → Update INPUT_PATH at the top of this script.")
        return

    print(f"\n  Loading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"  Loaded {len(candidates):,} candidates.")

    kept, removed = remove_twins(candidates)

    print(f"  Writing unique candidates → {unique_path}")
    write_json_array(unique_path, kept)

    print(f"  Writing removed twins     → {twins_path}")
    write_json_array(twins_path, removed)

    removal_rate = len(removed) / len(candidates) * 100 if candidates else 0

    print(f"""
╔══════════════════════════════════════════════════════╗
  TWIN REMOVAL COMPLETE
╠══════════════════════════════════════════════════════╣
  Input candidates    : {len(candidates):,}
  ──────────────────────────────────────────────────
  ✅ Unique kept      : {len(kept):,}
  👥 Twins removed    : {len(removed):,}  ({removal_rate:.1f}%)
  ──────────────────────────────────────────────────
  Threshold used      : {SIMILARITY_THRESHOLD}
  MinHash perms       : {NUM_PERMUTATIONS}
  ──────────────────────────────────────────────────
  Output files:
    {unique_path}
    {twins_path}
  ──────────────────────────────────────────────────
  If removal rate > 15%: raise threshold to 0.90
  If removal rate < 2% : lower threshold to 0.80
╚══════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    run()