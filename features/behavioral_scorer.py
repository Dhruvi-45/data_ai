import json
import re
from collections import defaultdict
from difflib import SequenceMatcher

def load_jsonl(path):
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
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
    This is fast (exact match) and catches the majority of twins in your data.
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
    PASS 2: For candidates not caught by Pass 1,
    use weighted fuzzy similarity score.
    """
    # Summary similarity (35%)
    s1 = c1["profile"].get("summary", "")
    s2 = c2["profile"].get("summary", "")
    summary_sim = text_similarity(s1, s2)

    # Career description blob similarity (40%)
    desc1 = " ".join(sorted([
        normalize_text(j.get("description",""))
        for j in c1.get("career_history",[])
    ]))
    desc2 = " ".join(sorted([
        normalize_text(j.get("description",""))
        for j in c2.get("career_history",[])
    ]))
    career_sim = text_similarity(desc1, desc2)

    # Skills Jaccard (15%)
    sk1 = set(s["name"].lower() for s in c1.get("skills", []))
    sk2 = set(s["name"].lower() for s in c2.get("skills", []))
    skills_sim = len(sk1 & sk2) / len(sk1 | sk2) if (sk1 | sk2) else 0

    # Education match (10%)
    edu1 = set((e.get("institution","").lower(), e.get("degree","").lower())
               for e in c1.get("education", []))
    edu2 = set((e.get("institution","").lower(), e.get("degree","").lower())
               for e in c2.get("education", []))
    edu_sim = 1.0 if edu1 == edu2 else 0.0

    total = (summary_sim * 0.35 + career_sim * 0.40 +
             skills_sim * 0.15 + edu_sim * 0.10)

    return total

def best_in_group(group):
    """Keep candidate with highest profile completeness score"""
    return max(group, key=lambda c: c["redrob_signals"]["profile_completeness_score"])

def remove_twins_two_pass(candidates, fuzzy_threshold=0.85):

    # ─────────────────────────────────────────
    # PASS 1: Fast exact description fingerprint
    # Catches the bulk of twins in your dataset
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
    # PASS 2: Fuzzy similarity on survivors
    # Catches near-twins that slightly altered descriptions
    # ─────────────────────────────────────────
    print(f"=== PASS 2: Fuzzy Similarity (threshold={fuzzy_threshold}) ===")
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

            score = compute_similarity_score(c1, c2)

            if score >= fuzzy_threshold:
                id1 = c1["candidate_id"]
                id2 = c2["candidate_id"]
                sc1 = c1["redrob_signals"]["profile_completeness_score"]
                sc2 = c2["redrob_signals"]["profile_completeness_score"]

                keep = id1 if sc1 >= sc2 else id2
                remove_id = id2 if keep == id1 else id1
                to_remove.add(remove_id)

                twin_pairs.append({"keep": keep, "remove": remove_id,
                                   "score": round(score, 3)})
                print(f"Pass2 twin (score={score:.3f}): {id1} vs {id2} "
                      f"→ keeping {keep}")

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
# --- Run ---  ✅ FIXED PATHS FOR COLAB
# ─────────────────────────────────────────
candidates = load_jsonl("candidates.jsonl")           # ✅ root level
final, twin_pairs = remove_twins_two_pass(candidates, fuzzy_threshold=0.85)

with open("candidates_no_twins.jsonl", "w", encoding="utf-8") as f:  # ✅ root level
    for c in final:
        f.write(json.dumps(c) + "\n")

with open("twin_report.json", "w") as f:              # ✅ root level
    json.dump(twin_pairs, f, indent=2)

print("\nDone. Saved candidates_no_twins.jsonl + twin_report.json")