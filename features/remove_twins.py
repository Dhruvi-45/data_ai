import json
import re
from collections import defaultdict
from difflib import SequenceMatcher

# ─────────────────────────────────────────
# STEP 1: Load directly from standard .json array
# ─────────────────────────────────────────
def load_json_array(path):
    with open(path, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    print(f"✅ Loaded {len(candidates)} candidates successfully.\n")
    return candidates

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'\d+\.?\d*\+?\s*(years?|yrs?)', 'X years', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def get_description_fingerprint(candidate):
    """
    PASS 1: Hash ALL job descriptions sorted and joined.
    If two candidates share the same description blob -> definite twins.
    Fast O(n) - catches majority of structural twins instantly.
    """
    descriptions = sorted([
        normalize_text(job.get("description", ""))
        for job in candidate.get("career_history", [])
        if job.get("description", "")
    ])

    # If no descriptions exist, return a unique ID token
    # so candidates with blank job histories aren't accidentally wiped out together.
    if not descriptions:
        return (candidate.get("candidate_id"),)

    return tuple(descriptions)

def text_similarity(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def compute_similarity_score(c1, c2):
    """
    PASS 2: Filter ONLY on remaining edges:
    - Summary text similarity    -> 35% weight
    - Job description similarity -> 65% weight (most important)
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

    total = (summary_sim * 0.35) + (career_sim * 0.65)
    return total, summary_sim, career_sim

def best_in_group(group):
    """Keep candidate with highest profile completeness score"""
    return max(group, key=lambda c: c["redrob_signals"]["profile_completeness_score"])

def remove_twins_two_pass(candidates, fuzzy_threshold=0.85):
    # Track the records categorized as twins
    twins_removed_records = []

    # ─────────────────────────────────────
    # PASS 1: Fast exact fingerprint
    # ─────────────────────────────────────
    print("=== PASS 1: Description Fingerprint (Fast) ===")
    desc_groups = defaultdict(list)
    for c in candidates:
        fp = get_description_fingerprint(c)
        desc_groups[fp].append(c)

    pass1_kept = []
    pass1_removed_count = 0
    for fp, group in desc_groups.items():
        chosen_one = best_in_group(group)
        pass1_kept.append(chosen_one)

        if len(group) > 1:
            print(f"Pass1 twin group ({len(group)}): {[c['candidate_id'] for c in group]}")
            pass1_removed_count += len(group) - 1
            # Add all the items we are skipping to the twins bucket
            for c in group:
                if c["candidate_id"] != chosen_one["candidate_id"]:
                    twins_removed_records.append(c)

    print(f"Pass 1 removed: {pass1_removed_count} twins")
    print(f"Remaining after Pass 1: {len(pass1_kept)}\n")

    # ─────────────────────────────────────
    # PASS 2: Summary + Job Description Fuzzy Analysis
    # ─────────────────────────────────────
    print(f"=== PASS 2: Summary + Job Description Only (threshold={fuzzy_threshold}) ===")
    n = len(pass1_kept)
    print(f"Comparing {n} candidates ({n*(n-1)//2} pairs)...")

    to_remove_ids = set()

    for i in range(n):
        for j in range(i + 1, n):
            c1 = pass1_kept[i]
            c2 = pass1_kept[j]

            if c1["candidate_id"] in to_remove_ids or c2["candidate_id"] in to_remove_ids:
                continue

            score, summary_sim, career_sim = compute_similarity_score(c1, c2)

            if score >= fuzzy_threshold:
                id1 = c1["candidate_id"]
                id2 = c2["candidate_id"]
                sc1 = c1["redrob_signals"]["profile_completeness_score"]
                sc2 = c2["redrob_signals"]["profile_completeness_score"]

                if sc1 >= sc2:
                    keep_cand, remove_cand = c1, c2
                else:
                    keep_cand, remove_cand = c2, c1

                to_remove_ids.add(remove_cand["candidate_id"])
                twins_removed_records.append(remove_cand)

                print(f"Twin found (total={score:.3f}) -> keeping {keep_cand['candidate_id']}, removing {remove_cand['candidate_id']}")

    final_kept = [c for c in pass1_kept if c["candidate_id"] not in to_remove_ids]

    print(f"\n" + '=' * 45)
    print(f"Original Pool       : {len(candidates)}")
    print(f"Clean Unique Saved  : {len(final_kept)}")
    print(f"Total Twins Removed : {len(twins_removed_records)}")
    print('=' * 45)

    return final_kept, twins_removed_records


# ─────────────────────────────────────────
# --- Execution Hook ---
# ─────────────────────────────────────────
# 1. Read input data from your exact file name
input_file = "againhoneyfiltered_candidates (1).json"
unique_list, twins_list = remove_twins_two_pass(load_json_array(input_file), fuzzy_threshold=0.85)

# 2. Write the Clean Unique Candidates File
with open("cleaned_candidates.json", "w", encoding="utf-8") as f:
    json.dump(unique_list, f, indent=2, ensure_ascii=False)

# 3. Write the Removed Twins Dataset File
with open("removed_twins.json", "w", encoding="utf-8") as f:
    json.dump(twins_list, f, indent=2, ensure_ascii=False)

print("\n🚀 Execution Complete. Saved 'twins_cleaned_candidates.json' & 'removed_twins.json'")