"""
Google Colab Honeypot Filter

Input:
    kept_candidates.json

Outputs:
    filtered_candidates.json
    honeypot_candidates.json
"""

import json
from pathlib import Path


# ============================================================================
# Detection helpers
# ============================================================================

def check_inverted_salary(candidate):
    flags = []
    try:
        sal = candidate["redrob_signals"]["expected_salary_range_inr_lpa"]
        mn, mx = sal.get("min", 0), sal.get("max", 0)

        if mn > mx:
            flags.append(
                f"INVERTED_SALARY: min={mn} LPA > max={mx} LPA"
            )

    except (KeyError, TypeError):
        pass

    return flags


def check_education_overlap(candidate):
    flags = []

    edu = candidate.get("education", [])

    for i in range(len(edu)):
        for j in range(i + 1, len(edu)):

            e1 = edu[i]
            e2 = edu[j]

            s1 = e1.get("start_year", 0)
            e1_end = e1.get("end_year", 0)

            s2 = e2.get("start_year", 0)
            e2_end = e2.get("end_year", 0)

            if s1 < e2_end and s2 < e1_end:
                flags.append(
                    f"EDU_OVERLAP: "
                    f"{e1.get('degree','?')} ({s1}-{e1_end}) overlaps "
                    f"{e2.get('degree','?')} ({s2}-{e2_end})"
                )

    return flags


_MECH_ENG_KEYWORDS = [
    "solidworks",
    "creo",
    "dfma",
    "dfm/",
    "fea (ansys)",
    "ansys",
    "product subsystems",
    "production tooling",
    "prototype, production",
    "hardware-development cadence",
]

_ENG_TITLE_FRAGMENTS = [
    "engineer",
    "mechanical",
    "hardware"
]


def check_desc_title_mismatch(candidate):
    flags = []

    for job in candidate.get("career_history", []):

        desc = str(job.get("description", "")).lower()
        title = str(job.get("title", "")).lower()

        has_mech = any(k in desc for k in _MECH_ENG_KEYWORDS)
        is_eng = any(k in title for k in _ENG_TITLE_FRAGMENTS)

        if has_mech and not is_eng:

            flags.append(
                f"DESC_TITLE_MISMATCH: "
                f"{job.get('title','?')} at "
                f"{job.get('company','?')}"
            )

            break

    return flags


def check_skill_exceeds_career(candidate):
    flags = []

    career_months = sum(
        j.get("duration_months", 0)
        for j in candidate.get("career_history", [])
    )

    for skill in candidate.get("skills", []):

        skill_months = skill.get("duration_months", 0)

        if skill_months > career_months + 3:

            flags.append(
                f"SKILL_EXCEEDS_CAREER: "
                f"{skill.get('name','?')} "
                f"{skill_months} months > "
                f"career {career_months} months"
            )

    return flags


# ============================================================================
# Classifier
# ============================================================================

ALL_CHECKS = [
    check_inverted_salary,
    check_education_overlap,
    check_desc_title_mismatch,
    check_skill_exceeds_career,
]


def classify_candidate(candidate):

    all_flags = []

    for check in ALL_CHECKS:
        all_flags.extend(check(candidate))

    return bool(all_flags), all_flags


# ============================================================================
# File Helpers
# ============================================================================

def load_candidates(filepath):

    text = Path(filepath).read_text(
        encoding="utf-8"
    ).strip()

    try:
        data = json.loads(text)

        if isinstance(data, list):
            return data

        return [data]

    except json.JSONDecodeError:

        return [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]


def save_json(data, filepath):

    Path(filepath).write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================================
# Main
# ============================================================================

INPUT_FILE = "kept_candidates.json"

CLEAN_OUTPUT = "filtered_candidates.json"
HONEYPOT_OUTPUT = "honeypot_candidates.json"

print("=" * 60)
print("HONEYPOT FILTER")
print("=" * 60)

candidates = load_candidates(INPUT_FILE)

print(f"Loaded {len(candidates)} candidates")

clean_candidates = []
honeypot_candidates = []

for candidate in candidates:

    is_honeypot, reasons = classify_candidate(candidate)

    if is_honeypot:

        candidate["_honeypot_reasons"] = reasons
        honeypot_candidates.append(candidate)

    else:

        clean_candidates.append(candidate)

save_json(
    clean_candidates,
    CLEAN_OUTPUT
)

save_json(
    honeypot_candidates,
    HONEYPOT_OUTPUT
)

total = len(candidates)
clean_count = len(clean_candidates)
honeypot_count = len(honeypot_candidates)

honeypot_rate = (
    honeypot_count / total * 100
    if total else 0
)

print("\nSUMMARY")
print("-" * 60)
print(f"Total Candidates : {total}")
print(f"Filtered (Clean) : {clean_count}")
print(f"Honeypots        : {honeypot_count}")
print(f"Honeypot Rate    : {honeypot_rate:.2f}%")
print("-" * 60)

print("\nFiles Generated:")
print(f"✓ {CLEAN_OUTPUT}")
print(f"✓ {HONEYPOT_OUTPUT}")

# Breakdown
reason_counts = {}

for c in honeypot_candidates:

    for r in c.get("_honeypot_reasons", []):

        key = r.split(":")[0]

        reason_counts[key] = (
            reason_counts.get(key, 0) + 1
        )

if reason_counts:

    print("\nHoneypot Breakdown:")

    for k, v in sorted(
        reason_counts.items(),
        key=lambda x: -x[1]
    ):
        print(f"{k}: {v}")