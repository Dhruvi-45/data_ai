"""
Redrob Hackathon — JD-aware candidate embedding & ranking
===========================================================

How this differs from the starter script you shared:

1. THE JD IS PARSED, NOT HAND-TYPED.
   The old script's `get_structured_jd_text()` was a human-written summary
   of the JD. That means every time the JD changes, a person has to notice
   and rewrite the summary, and any nuance the person forgot to copy is
   silently lost. This script reads the actual .docx, walks it by heading
   ("Things you absolutely need", "Things we explicitly do NOT want", etc.)
   and builds the semantic text straight from that — so it tracks the doc.

2. THE JD ISN'T ONE VECTOR, IT'S THREE.
   This particular JD is explicit that a single "does this look similar to
   the JD" score is a trap: a candidate can be full of the right nouns
   (RAG, Pinecone, embeddings...) and still be wrong (e.g. a Marketing
   Manager, a pure-research career, a pure-consulting career, a computer
   vision specialist with no NLP/IR background). So three JD vectors are
   built — MUST-HAVE, NICE-TO-HAVE, and EXCLUDE (the "do not want" /
   disqualifier text) — and a candidate is rewarded for similarity to the
   first two and *penalized* for similarity to the third.

3. CANDIDATE VECTORS COME FROM CAREER HISTORY, NOT THE SKILLS LIST.
   Kept from your version on purpose: a self-reported skills array is the
   easiest thing to stuff with keywords, so it's excluded from the
   embedding text. It's still used (in `compute_red_flags`) for
   non-embedding rule checks, just not allowed to dominate the vector.

4. AVAILABILITY IS A SEPARATE MULTIPLIER, NOT PART OF THE TEXT.
   The JD's hackathon note says a perfect-on-paper candidate who hasn't
   logged in for months and never responds to recruiters should be
   down-weighted. That's a behavioral fact, not a semantic one, so it's
   computed from `redrob_signals` and applied as a multiplier on top of
   the embedding score rather than mixed into the text.

Install once: pip install python-docx sentence-transformers --break-system-packages
Run:          python redrob_match.py --jd job_description.docx --candidates candidates.json
"""

import argparse
import json
import math
import re
from datetime import date, datetime

import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────────────────────────────────
# CONFIG — tune scoring behaviour here without touching the logic below
# ──────────────────────────────────────────────────────────────────────────
CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "jd_path": "job_description.docx",
    "candidates_path": "sample_candidates.json",
    "output_path": "ranked_candidates.json",
    "weight_must_have": 1.0,        # how much MUST-HAVE similarity counts
    "weight_nice_to_have": 0.35,    # how much NICE-TO-HAVE similarity counts
    "weight_exclude": 0.9,          # how hard EXCLUDE similarity is punished
    "red_flag_penalty": 0.12,       # flat deduction per rule-based red flag
    "availability_floor": 0.55,     # a 0-availability candidate still keeps this fraction of their semantic score
    "recency_halflife_days": 60,    # "last active" recency decay half-life
}

# Heuristics for the rule-based red flags (see compute_red_flags). These are
# deliberately simple substring checks — they're a cheap second opinion next
# to the EXCLUDE embedding, not a replacement for it.
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "mindtree", "ltimindtree", "hcl", "hcltech",
    "tech mahindra", "genpact", "mphasis", "l&t infotech",
}
CV_SPEECH_ROBOTICS_HINTS = {
    "computer vision", "image classification", "object detection",
    "speech recognition", "speech-to-text", "text-to-speech", "tts", "asr",
    "robotics", "autonomous",
}
IR_NLP_HINTS = {
    "nlp", "retrieval", "search", "ranking", "rank", "recommendation",
    "embeddings", "information retrieval", "re-ranking", "rerank",
    "vector database", "matching engine", "hybrid search",
}
RESEARCH_INDUSTRY_HINTS = {"research", "academia", "higher education", "university"}


def clean(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


# ──────────────────────────────────────────────────────────────────────────
# STEP 1 — Read the JD straight out of the .docx, by heading
# ──────────────────────────────────────────────────────────────────────────
def parse_jd_sections(docx_path):
    """Walk the document paragraph-by-paragraph, bucketing text under
    whatever heading it sits beneath. Works for any heading wording —
    the mapping to semantic buckets happens separately in build_jd_blocks."""
    doc = Document(docx_path)
    sections, current = {}, "_preamble"
    sections[current] = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        if p.style.name.startswith("Heading"):
            current = text
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(text)
    return sections


def _find_heading(sections, *needles):
    """Fuzzy-match a heading by requiring all `needles` as substrings
    (case-insensitive), so small copy edits to the JD don't break parsing."""
    for heading in sections:
        low = heading.lower()
        if all(n in low for n in needles):
            return heading
    return None


def build_jd_blocks(sections):
    """Map this doc's actual headings onto the four buckets we score against."""
    must_h = _find_heading(sections, "absolutely need")
    nice_h = _find_heading(sections, "won't reject")
    exclude_h = _find_heading(sections, "not want")
    disqualifiers_h = _find_heading(sections, "mean by")  # "What we mean by 5-9 years" holds the disqualifier bullets
    ideal_h = _find_heading(sections, "between the lines")
    role_h = _find_heading(sections, "actually be doing")
    honest_h = _find_heading(sections, "honest about this role")

    def grab(*heads):
        out = []
        for h in heads:
            if h:
                out.extend(sections.get(h, []))
        return clean(" ".join(out))

    blocks = {
        "must_have": grab(must_h),
        "nice_to_have": grab(nice_h),
        "exclude": grab(exclude_h, disqualifiers_h),
        "ideal_profile": grab(ideal_h, role_h, honest_h, "_preamble"),
    }

    for name, text in blocks.items():
        if not text:
            print(f"  ⚠ Could not find content for JD block '{name}' — "
                  f"check that the heading wording in the docx still matches.")
    return blocks


# ──────────────────────────────────────────────────────────────────────────
# STEP 2 — Turn each candidate into career-history-driven prose
# ──────────────────────────────────────────────────────────────────────────
def format_candidate_narrative(candidate):
    """Same anti-honeypot principle as your original script: build the
    embedding text from what they actually did (career history + summary),
    not from the self-reported skills list, so keyword-stuffed skills can't
    out-rank genuine builders."""
    profile = candidate.get("profile", {}) or {}
    history = candidate.get("career_history", []) or []

    career_bits = []
    for job in history:
        job = job or {}
        desc = job.get("description")
        if not desc:
            continue
        career_bits.append(
            f"As a {job.get('title', 'professional')} in {job.get('industry', 'industry')}: {desc}"
        )

    narrative = (
        f"Professional title: {profile.get('current_title', '')}. "
        f"Industry: {profile.get('current_industry', '')}. "
        f"Summary: {profile.get('summary', '')}. "
        f"Career history: {' '.join(career_bits)}"
    )
    return clean(narrative)


# ──────────────────────────────────────────────────────────────────────────
# STEP 3 — Rule-based red flags (cheap second opinion next to EXCLUDE embedding)
# ──────────────────────────────────────────────────────────────────────────
def compute_red_flags(candidate):
    profile = candidate.get("profile", {}) or {}
    history = candidate.get("career_history", []) or []

    companies = [clean(profile.get("current_company", ""))]
    companies += [clean(h.get("company", "")) for h in history if h]
    companies = [c for c in companies if c]

    industries = [clean(profile.get("current_industry", ""))]
    industries += [clean(h.get("industry", "")) for h in history if h]
    industries = [i for i in industries if i]

    flags = []

    if companies and all(any(firm in c for firm in CONSULTING_FIRMS) for c in companies):
        flags.append("pure_consulting_career")

    if industries and all(any(r in i for r in RESEARCH_INDUSTRY_HINTS) for i in industries):
        flags.append("pure_research_no_production")

    text_blob = clean(
        profile.get("summary", "") + " " + profile.get("current_title", "") + " "
        + " ".join((h or {}).get("description", "") or "" for h in history) + " "
        + " ".join(s.get("name", "") for s in (candidate.get("skills") or []))
    )
    looks_cv_speech_robotics = any(h in text_blob for h in CV_SPEECH_ROBOTICS_HINTS)
    looks_ir_nlp = any(h in text_blob for h in IR_NLP_HINTS)
    if looks_cv_speech_robotics and not looks_ir_nlp:
        flags.append("cv_speech_robotics_without_nlp_ir")

    return flags


# ──────────────────────────────────────────────────────────────────────────
# STEP 4 — Availability multiplier from behavioral signals
# ──────────────────────────────────────────────────────────────────────────
def compute_availability(candidate, halflife_days=60, reference_date=None):
    signals = candidate.get("redrob_signals", {}) or {}
    reference_date = reference_date or date.today()

    recency = 0.5  # neutral default if we can't parse a date
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_ago = max((reference_date - last_dt).days, 0)
            recency = math.exp(-math.log(2) * days_ago / halflife_days)
        except ValueError:
            pass

    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.4

    response_rate = signals.get("recruiter_response_rate")
    response_rate = response_rate if isinstance(response_rate, (int, float)) else 0.3

    availability = 0.45 * recency + 0.25 * open_to_work + 0.30 * response_rate
    return max(0.0, min(1.0, availability))


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def cosine_sim_batch(matrix, vector):
    """matrix: (N, D) candidate vectors, vector: (D,) a single JD vector."""
    matrix_n = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    vector_n = vector / (np.linalg.norm(vector) + 1e-9)
    return matrix_n @ vector_n


# ──────────────────────────────────────────────────────────────────────────
# STEP 5 — Pipeline
# ──────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg = {**CONFIG, "jd_path": args.jd, "candidates_path": args.candidates, "output_path": args.out}

    print(f"🔄 Loading embedding model ({cfg['model_name']}) ...")
    model = SentenceTransformer(cfg["model_name"])
    print("✅ Model loaded.")

    print(f"\n📄 Parsing JD from {cfg['jd_path']} ...")
    sections = parse_jd_sections(cfg["jd_path"])
    blocks = build_jd_blocks(sections)

    core_text = clean(blocks["must_have"] + " " + blocks["ideal_profile"])
    nice_text = blocks["nice_to_have"]
    exclude_text = blocks["exclude"]

    jd_vecs = model.encode([core_text, nice_text, exclude_text])
    core_vec, nice_vec, exclude_vec = jd_vecs
    print(f"🎯 JD vectors built — MUST-HAVE/IDEAL, NICE-TO-HAVE, EXCLUDE. Dim: {core_vec.shape[0]}")

    print(f"\n📦 Loading candidates from {cfg['candidates_path']} ...")
    with open(cfg["candidates_path"], encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"   {len(candidates)} candidates loaded.")

    narratives = [format_candidate_narrative(c) for c in candidates]
    cand_vecs = np.array(model.encode(narratives, batch_size=32, show_progress_bar=False))
    print(f"✨ Vectorized {len(candidates)} candidates.")

    sim_core = cosine_sim_batch(cand_vecs, core_vec)
    sim_nice = cosine_sim_batch(cand_vecs, nice_vec)
    sim_exclude = cosine_sim_batch(cand_vecs, exclude_vec)

    weight_total = cfg["weight_must_have"] + cfg["weight_nice_to_have"]

    results = []
    for i, c in enumerate(candidates):
        flags = compute_red_flags(c)
        availability = compute_availability(c, cfg["recency_halflife_days"])

        semantic = (
            cfg["weight_must_have"] * sim_core[i] + cfg["weight_nice_to_have"] * sim_nice[i]
        ) / weight_total
        semantic -= cfg["weight_exclude"] * sim_exclude[i]
        semantic -= cfg["red_flag_penalty"] * len(flags)

        availability_multiplier = cfg["availability_floor"] + (1 - cfg["availability_floor"]) * availability
        final_score = semantic * availability_multiplier

        profile = c.get("profile", {}) or {}
        results.append({
            "candidate_id": c.get("candidate_id"),
            "name": profile.get("anonymized_name"),
            "current_title": profile.get("current_title"),
            "years_experience": profile.get("years_of_experience"),
            "sim_must_have": round(float(sim_core[i]), 4),
            "sim_nice_to_have": round(float(sim_nice[i]), 4),
            "sim_exclude": round(float(sim_exclude[i]), 4),
            "red_flags": flags,
            "availability_score": round(availability, 3),
            "final_score": round(float(final_score), 4),
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    with open(cfg["output_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n📝 Full ranking written to {cfg['output_path']}")
    print(f"\nTop {args.top} matches:")
    for r in results[: args.top]:
        flag_str = f"  ⚠ {','.join(r['red_flags'])}" if r["red_flags"] else ""
        print(f"  #{r['rank']:>3}  {r['final_score']:>7.3f}  {r['name']:<22} {r['current_title']}{flag_str}")


def parse_args():
    p = argparse.ArgumentParser(description="Embed a Redrob JD + candidate pool and rank by semantic fit.")
    p.add_argument("--jd", default=CONFIG["jd_path"], help="Path to the JD .docx")
    p.add_argument("--candidates", default=CONFIG["candidates_path"], help="Path to the candidates .json")
    p.add_argument("--out", default=CONFIG["output_path"], help="Where to write the ranked results")
    p.add_argument("--top", type=int, default=10, help="How many top matches to print to console")
    return p.parse_args()


if __name__ == "__main__":
    main()