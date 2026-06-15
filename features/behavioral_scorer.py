import gzip
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher

# ─────────────────────────────────────────
# STEP 1: Load directly from .gz file
# No need to unzip manually!
# ─────────────────────────────────────────
def load_jsonl(path):
    candidates = []
    skipped = 0

    # Handle both .gz and plain .jsonl files
    if path.endswith(".gz"):
        open_func = gzip.open
        mode = "rt"               # rt = read text mode for gzip
    else:
        open_func = open
        mode = "r"

    with open_func(path, mode, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as e:
                skipped += 1
                print(f"⚠️  Skipping line {line_num} — JSON error: {e}")

    print(f"✅ Loaded {len(candidates)} candidates "
          f"({skipped} bad lines skipped)\n")
    return candidates

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'\d+\.?\d*\+?\s*(years?|yrs?)', 'X years', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def get_description_fingerprint(candidate):
    """
    PASS 1: Hash ALL job descriptions sorted and joined.
    If two candidates share the same description blob → definite twins.
    Fast O(n) — catches majority of twins instantly.
    """
    descriptions = sorted([
        normalize_text(job.get("description", ""))
        for job in candidate.get("career_history", [])
        if job.get("description", "")
    ])
    return tuple(descriptions)

def text_similarity(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def compute_similarity_score(c1, c2):
    """
    PASS 2: Filter ONLY on:
    - Summary text similarity    → 35% weight
    - Job description similarity → 65% weight (most important)
    """
    # Summary similarity (35%)
    s1 = c1["profile"].get("summary", "")
    s2 = c2["profile"].get("summary", "")
    summary_sim = text_similarity(s1, s2)

    # Career description blob similarity (65%)
    desc1 = " ".join(sorted([
        normalize_text(j.get("description", ""))
        for j in c1.get("career_history", [])
    ]))
    desc2 = " ".join(sorted([
        normalize_text(j.get("description", ""))
        for j in c2.get("career_history", [])
    ]))
    career_sim = text_similarity(desc1, desc2)

    # Summary 35% + Career 65% = 100%
    total = (summary_sim * 0.35) + (career_sim * 0.65)

    return total, summary_sim, career_sim

def best_in_group(group):
    """Keep candidate with highest profile completeness score"""
    return max(group, key=lambda c: c["redrob_signals"]["profile_completeness_score"])

def remove_twins_two_pass(candidates, fuzzy_threshold=0.85):

    # ─────────────────────────────────────────
    # PASS 1: Fast exact description fingerprint
    # ─────────────────────────────────────────
    print("=== PASS 1: Description Fingerprint (Fast) ===")
    desc_groups = defaultdict(list)
    for c in candidates:
        fp = get_description_fingerprint(c)
        desc_groups[fp].append(c)

    pass1_kept = []
    pass1_removed = 0
    for fp, group in desc_groups.items():
        if len(group) > 1:
            print(f"Pass1 twin group ({len(group)}): "
                  f"{[c['candidate_id'] for c in group]}")
            pass1_removed += len(group) - 1
        pass1_kept.append(best_in_group(group))

    print(f"Pass 1 removed: {pass1_removed} twins")
    print(f"Remaining after Pass 1: {len(pass1_kept)}\n")

    # ─────────────────────────────────────────
    # PASS 2: Summary + Job Description Only
    # Summary  → 35%
    # Job Desc → 65%
    # ─────────────────────────────────────────
    print(f"=== PASS 2: Summary + Job Description Only "
          f"(threshold={fuzzy_threshold}) ===")
    print(f"  Summary similarity   → 35% of score")
    print(f"  Job desc similarity  → 65% of score ← most important\n")

    n = len(pass1_kept)
    print(f"Comparing {n} candidates ({n*(n-1)//2} pairs)...")

    to_remove = set()
    twin_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            c1 = pass1_kept[i]
            c2 = pass1_kept[j]

            if c1["candidate_id"] in to_remove:
                continue

            score, summary_sim, career_sim = compute_similarity_score(c1, c2)

            if score >= fuzzy_threshold:
                id1 = c1["candidate_id"]
                id2 = c2["candidate_id"]
                sc1 = c1["redrob_signals"]["profile_completeness_score"]
                sc2 = c2["redrob_signals"]["profile_completeness_score"]

                keep = id1 if sc1 >= sc2 else id2
                remove_id = id2 if keep == id1 else id1
                to_remove.add(remove_id)

                twin_pairs.append({
                    "keep": keep,
                    "remove": remove_id,
                    "total_score": round(score, 3),
                    "summary_similarity": round(summary_sim, 3),
                    "career_similarity": round(career_sim, 3)
                })

                print(f"Twin found (total={score:.3f}) "
                      f"[summary={summary_sim:.3f}, "
                      f"career={career_sim:.3f}]: "
                      f"{id1} vs {id2} → keeping {keep}")

    final_kept = [c for c in pass1_kept
                  if c["candidate_id"] not in to_remove]

    print(f"\nPass 2 removed: {len(to_remove)} more twins")
    print(f"{'='*45}")
    print(f"Original : {len(candidates)}")
    print(f"After P1 : {len(pass1_kept)}")
    print(f"Final    : {len(final_kept)}")
    print(f"Total removed: {len(candidates) - len(final_kept)}")

    return final_kept, twin_pairs


# ─────────────────────────────────────────
# --- Run ---
# Upload candidates.jsonl.gz to Colab
# then run this
# ─────────────────────────────────────────
candidates = load_jsonl("candidates.jsonl.gz")   # ✅ load directly from .gz
final, twin_pairs = remove_twins_two_pass(candidates, fuzzy_threshold=0.85)

with open("candidates_no_twins.jsonl", "w", encoding="utf-8") as f:
    for c in final:
        f.write(json.dumps(c) + "\n")

with open("twin_report.json", "w") as f:
    json.dump(twin_pairs, f, indent=2)

print("\nDone. Saved candidates_no_twins.jsonl + twin_report.json")






import gzip
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher

def load_jsonl(path):
    candidates = []
    skipped = 0

    if path.endswith(".gz"):
        open_func = gzip.open
        mode = "rt"
    else:
        open_func = open
        mode = "r"

    with open_func(path, mode, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as e:
                skipped += 1
                print(f"⚠️  Skipping line {line_num} — JSON error: {e}")

    print(f"✅ Loaded {len(candidates)} candidates "
          f"({skipped} bad lines skipped)\n")
    return candidates

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'\d+\.?\d*\+?\s*(years?|yrs?)', 'X years', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def get_description_fingerprint(candidate):
    descriptions = sorted([
        normalize_text(job.get("description", ""))
        for job in candidate.get("career_history", [])
        if job.get("description", "")
    ])
    return tuple(descriptions)

def text_similarity(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def compute_similarity_score(c1, c2):
    # Summary similarity (35%)
    s1 = c1["profile"].get("summary", "")
    s2 = c2["profile"].get("summary", "")
    summary_sim = text_similarity(s1, s2)

    # Career description blob similarity (65%)
    desc1 = " ".join(sorted([
        normalize_text(j.get("description", ""))
        for j in c1.get("career_history", [])
    ]))
    desc2 = " ".join(sorted([
        normalize_text(j.get("description", ""))
        for j in c2.get("career_history", [])
    ]))
    career_sim = text_similarity(desc1, desc2)

    total = (summary_sim * 0.35) + (career_sim * 0.65)
    return total, summary_sim, career_sim

def best_in_group(group):
    return max(group, key=lambda c: c["redrob_signals"]["profile_completeness_score"])

def remove_twins_two_pass(candidates, fuzzy_threshold=0.85):

    # ─────────────────────────────────────
    # PASS 1: Fast exact fingerprint
    # ─────────────────────────────────────
    print("=== PASS 1: Description Fingerprint (Fast) ===")
    desc_groups = defaultdict(list)
    for c in candidates:
        fp = get_description_fingerprint(c)
        desc_groups[fp].append(c)

    pass1_kept = []
    pass1_removed = 0
    for fp, group in desc_groups.items():
        if len(group) > 1:
            print(f"Pass1 twin group ({len(group)}): "
                  f"{[c['candidate_id'] for c in group]}")
            pass1_removed += len(group) - 1
        pass1_kept.append(best_in_group(group))

    print(f"Pass 1 removed: {pass1_removed} twins")
    print(f"Remaining after Pass 1: {len(pass1_kept)}\n")

    # ─────────────────────────────────────
    # PASS 2: Summary + Job Description
    # Summary  → 35%
    # Job Desc → 65%
    # ─────────────────────────────────────
    print(f"=== PASS 2: Summary + Job Description Only "
          f"(threshold={fuzzy_threshold}) ===")
    print(f"  Summary similarity   → 35% of score")
    print(f"  Job desc similarity  → 65% of score ← most important\n")

    n = len(pass1_kept)
    print(f"Comparing {n} candidates ({n*(n-1)//2} pairs)...")

    to_remove = set()
    twin_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            c1 = pass1_kept[i]
            c2 = pass1_kept[j]

            # ✅ FIXED — skip if EITHER candidate already marked as twin
            if c1["candidate_id"] in to_remove:
                continue
            if c2["candidate_id"] in to_remove:
                continue

            score, summary_sim, career_sim = compute_similarity_score(c1, c2)

            if score >= fuzzy_threshold:
                id1 = c1["candidate_id"]
                id2 = c2["candidate_id"]
                sc1 = c1["redrob_signals"]["profile_completeness_score"]
                sc2 = c2["redrob_signals"]["profile_completeness_score"]

                keep    = id1 if sc1 >= sc2 else id2
                remove_id = id2 if keep == id1 else id1
                to_remove.add(remove_id)

                twin_pairs.append({
                    "keep": keep,
                    "remove": remove_id,
                    "total_score": round(score, 3),
                    "summary_similarity": round(summary_sim, 3),
                    "career_similarity": round(career_sim, 3)
                })

                print(f"Twin found (total={score:.3f}) "
                      f"[summary={summary_sim:.3f}, "
                      f"career={career_sim:.3f}]: "
                      f"{id1} vs {id2} → keeping {keep}")

    final_kept = [c for c in pass1_kept
                  if c["candidate_id"] not in to_remove]

    print(f"\nPass 2 removed: {len(to_remove)} more twins")
    print(f"{'='*45}")
    print(f"Original  : {len(candidates)}")
    print(f"After P1  : {len(pass1_kept)}")
    print(f"Final     : {len(final_kept)}")
    print(f"Total removed: {len(candidates) - len(final_kept)}")

    return final_kept, twin_pairs


# ─────────────────────────────────────────
# --- Run ---
# ─────────────────────────────────────────
candidates = load_jsonl("candidates.jsonl.gz")
final, twin_pairs = remove_twins_two_pass(candidates, fuzzy_threshold=0.85)

with open("candidates_no_twins.jsonl", "w", encoding="utf-8") as f:
    for c in final:
        f.write(json.dumps(c) + "\n")

with open("twin_report.json", "w") as f:
    json.dump(twin_pairs, f, indent=2)

print("\nDone. Saved candidates_no_twins.jsonl + twin_report.json")