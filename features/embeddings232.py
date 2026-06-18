"""
score_candidates.py
====================
Stage 5 of the Redrob AI pipeline: Trap-resistant candidate scoring.

WHY THIS STAGE EXISTS
──────────────────────
embed_pipeline.py (stage 4) produces ONE embedding per candidate that
includes an explicit "Technical skills: X, Y, Z." line built by matching
the candidate's skills array against a keyword list. That line reads
almost identically to the JD's own requirements list — which means a
candidate who simply lists more AI buzzwords as "skills" gets a high
cosine similarity score regardless of whether they ever built anything.

The JD says this directly:

    "The 'right answer' to this JD is not 'find candidates whose skills
    section contains the most AI keywords.' That's a trap we've
    explicitly built into the dataset."

    "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in
    their profile, but if their career history shows they built a
    recommendation system at a product company, they're a fit. A
    candidate who has all the AI keywords listed as skills but whose
    title is 'Marketing Manager' is not a fit."

So this stage deliberately separates two things that stage 4 conflated:

  1. NARRATIVE SCORE (primary, trap-resistant)
     A fresh embedding built ONLY from career narrative — current role,
     job-to-job trajectory, actual job descriptions, education. The
     skills keyword line is excluded entirely. This is what drives the
     final ranking.

  2. KEYWORD SCORE (diagnostic only, never rewarded)
     The original "with skills" embedding from candidate_embeddings.npz.
     We compute the GAP between keyword_score and narrative_score. A
     large positive gap means "looks great on a skills list, doesn't
     look great in actual career narrative" — i.e. probable keyword
     stuffing — and gets flagged as a caution, never as a bonus.

On top of that we layer:

  - A shipper-vs-researcher tilt (semantic persona-anchor similarity +
    rule-based language signals from job descriptions), matching the
    JD's explicit ask: "we'd rather you tilt slightly toward shipper
    than toward researcher."
  - Hard/soft disqualifier detectors lifted directly from the JD's
    "disqualifiers we actually apply" and "things we explicitly do NOT
    want" sections (pure-research-only, LangChain-tutorial-only,
    architect drift, consulting-only career, CV/speech/robotics without
    NLP, title-chasing).
  - An availability multiplier for the JD's closing note about
    down-weighting inactive / low-response-rate candidates.

HOW TO USE IN GOOGLE COLAB
────────────────────────────
Run this AFTER embed_pipeline.py has produced:
    jd_embedding.npy, candidate_embeddings.npz, candidates_with_text.json

    !python score_candidates.py

CONFIG YOU MAY NEED TO ADJUST
───────────────────────────────
The "behavioral availability" section (last active date / recruiter
response rate) references field names I can't verify against your real
kept_candidates.json schema, since embed_pipeline.py never reads those
fields. Check LAST_ACTIVE_KEYS / RESPONSE_RATE_KEYS below and add your
actual key names if they differ — the script degrades gracefully
(no penalty, just a "no activity data found" note) if it can't find them.
"""

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ── CONFIGURE ────────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64

JD_EMBEDDING_PATH = "jd_embedding.npy"
CANDIDATE_EMBEDDINGS_PATH = "candidate_embeddings.npz"
CANDIDATES_WITH_TEXT_PATH = "candidates_with_text.json"

OUTPUT_RANKED_JSON = "ranked_candidates.json"
OUTPUT_SUMMARY_CSV = "ranked_candidates_summary.csv"

# Scoring weights — tune these, they are not sacred
SHIPPER_TILT_WEIGHT = 0.30       # how much shipper-tilt nudges the final score
SOFT_FLAG_PENALTY = 0.15         # multiplicative penalty per soft disqualifier
MIN_PENALTY_MULTIPLIER = 0.30    # floor so multiple soft flags don't zero a candidate
INACTIVITY_DAYS_SEVERE = 180     # JD explicitly mentions 6 months
INACTIVITY_DAYS_MILD = 90
LOW_RESPONSE_RATE_THRESHOLD = 0.10

# Field names for behavioral availability — ADJUST to match your real schema.
# Checked at both the top level of a candidate record and inside candidate["profile"].
LAST_ACTIVE_KEYS = [
    "last_active_at", "last_active", "last_login", "last_login_at",
    "last_seen", "platform_last_active",
]
RESPONSE_RATE_KEYS = [
    "recruiter_response_rate", "response_rate", "reply_rate",
    "recruiter_reply_rate",
]
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════
#  PART 1: PERSONA ANCHORS (shipper vs researcher)
#  Embedded once, then every candidate's narrative embedding is
#  compared against both. The relative distance is the semantic
#  half of the shipper/researcher tilt signal.
# ═══════════════════════════════════════════════════════════════

SHIPPER_ANCHOR_TEXT = """
Product engineer who ships fast under ambiguity. Took a working ranker
to real users in days, not months, starting from BM25 or simple
heuristics and layering embeddings and hybrid retrieval only once real
usage data justified the added complexity. Comfortable owning an
end-to-end system in production, iterating on recruiter or user
feedback loops, and treating the first version as deliberately good
enough rather than optimal. Cares more about shipped impact, A/B test
results, and engagement metrics than about novel model architectures.
""".strip()

RESEARCHER_ANCHOR_TEXT = """
Research-oriented engineer focused on model quality, novel
architectures, and academic rigor. Career centered on publishing
papers, running offline benchmarks, and pushing state-of-the-art
results on held-out datasets, with limited exposure to deploying
systems to real production users at scale. Prioritizes theoretical
correctness and experimental thoroughness over shipping speed, and is
more comfortable in research labs or pure-research roles than in
fast-moving product teams.
""".strip()


# ═══════════════════════════════════════════════════════════════
#  PART 2: LEXICONS
#  Rule-based signals used for tilt/disqualifier detection. None of
#  these are used to reward keyword density — they only ever gate
#  or nudge a score that's already anchored by the narrative
#  embedding (Part 4).
# ═══════════════════════════════════════════════════════════════

SHIPPER_PHRASES = [
    "shipped", "launched", "deployed to production", "in production",
    "real users", "live to users", "rolled out", "owned end-to-end",
    "end-to-end ownership", "mvp", "scrappy", "iterated", "a/b test",
    "ab test", "recruiter feedback", "production system", "at scale",
    "million users", "thousand users", "daily active users", "dau", "mau",
]

RESEARCHER_PHRASES = [
    "published", "publication", "paper accepted", "arxiv", "peer-reviewed",
    "conference paper", "novel architecture", "state-of-the-art", "sota",
    "phd thesis", "doctoral", "postdoc", "research fellow",
    "research scientist", "academic lab", "benchmark dataset", "theoretical",
]

PRODUCTION_DEPLOYMENT_PHRASES = [
    "production", "deployed", "shipped", "live", "real users",
    "scaled to", "rolled out to", "in prod",
]

SCALE_NUMBER_PATTERN = re.compile(
    r"\d[\d,\.]*\s*(million|mn|m\+|thousand|k\+|users|requests|qps|dau|mau)",
    re.IGNORECASE,
)

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
    "mphasis", "l&t infotech", "ltimindtree",
}

CV_SPEECH_ROBOTICS_TERMS = {
    "computer vision", "image classification", "object detection",
    "speech recognition", "asr ", " tts", "robotics", "slam",
    "autonomous vehicle",
}

NLP_IR_TERMS = {
    "nlp", "natural language processing", "information retrieval",
    "search", "ranking", "retrieval", "embedding", "text classification",
    "named entity", "language model",
}

PRE_LLM_IR_TERMS = {
    "bm25", "tf-idf", "elasticsearch", "solr", "lucene", "word2vec",
    "collaborative filtering", "information retrieval",
}

PURE_RESEARCH_TITLES = {
    "research scientist", "research fellow", "postdoc", "post-doctoral",
    "research intern", "phd candidate", "professor", "research associate",
}

ARCHITECT_DRIFT_TITLES = {
    "architect", "tech lead", "principal engineer", "engineering manager",
    "head of engineering", "director of engineering",
}

LANGCHAIN_TUTORIAL_TERMS = {
    "langchain", "openai api", "prompt engineering", "chatbot demo",
}

SENIORITY_ORDER = ["junior", "engineer", "senior", "staff", "principal", "distinguished"]

# Diagnostic-only lexical signals shown in the output for human reviewers.
# These NEVER affect final_score — they exist so a recruiter can sanity-check
# "did this person ever literally mention these technologies" without that
# check being what drives the ranking.
EMBEDDING_RETRIEVAL_TERMS = {
    "embedding", "embeddings", "sentence-transformers", "bge", "e5",
    "bert", "dense retrieval",
}
VECTOR_DB_TERMS = {
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "vector database", "vector db",
}
EVAL_FRAMEWORK_TERMS = {
    "ndcg", "mrr", "map", "a/b test", "ab test", "offline benchmark",
    "evaluation framework", "recruiter feedback",
}


# ═══════════════════════════════════════════════════════════════
#  PART 3: SMALL HELPERS
# ═══════════════════════════════════════════════════════════════

def _lower(text) -> str:
    return text.lower().strip() if text else ""

def _job_text(job: dict) -> str:
    return _lower(
        (job.get("title", "") or "") + " "
        + (job.get("description", "") or "") + " "
        + (job.get("company", "") or "")
    )

def _career_full_text(history: list) -> str:
    return " ".join(_job_text(j) for j in history)

def _count_phrase_hits(text: str, phrases) -> int:
    return sum(1 for p in phrases if p in text)

def _find_first_present(candidate: dict, key_variants: list):
    profile = candidate.get("profile", {}) or {}
    for key in key_variants:
        if candidate.get(key) not in (None, ""):
            return candidate[key]
        if profile.get(key) not in (None, ""):
            return profile[key]
    return None

def _seniority_rank(title: str) -> int:
    t = _lower(title)
    for i, kw in enumerate(SENIORITY_ORDER):
        if kw in t:
            return i
    return -1


# ═══════════════════════════════════════════════════════════════
#  PART 4: TRAP-RESISTANT NARRATIVE TEXT
#  Deliberately excludes the skills-keyword line. This is the ONLY
#  text used to compute the primary fit score (narrative_score).
# ═══════════════════════════════════════════════════════════════

def build_narrative_text(candidate: dict) -> str:
    profile = candidate.get("profile", {}) or {}
    history = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []

    parts = []

    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", "")
    if title:
        line = title
        if company:
            line += f" at {company}"
        if yoe:
            line += f" with {yoe} years of experience"
        parts.append(line + ".")

    headline = profile.get("headline", "")
    if headline and headline != title:
        parts.append(headline + ".")

    sorted_history = sorted(
        history, key=lambda j: j.get("start_date") or "0000", reverse=True
    )

    trajectory = []
    for job in sorted_history[:4]:
        t, c_, d = job.get("title", ""), job.get("company", ""), job.get("duration_months", 0) or 0
        if t and c_:
            trajectory.append(f"{t} at {c_} ({round(d / 12, 1)}y)")
    if trajectory:
        parts.append("Career: " + " -> ".join(trajectory) + ".")

    # Actual job descriptions carry the real signal — not the skills list.
    for job in sorted_history[:2]:
        desc = job.get("description", "")
        if desc:
            if len(desc) > 220:
                desc = desc[:220].rsplit(" ", 1)[0] + "..."
            parts.append(f"{job.get('title', '')}: {desc}")

    for edu in education[:1]:
        field = edu.get("field_of_study", "")
        degree = edu.get("degree", "")
        if field:
            parts.append(f"Education: {degree} in {field}.".strip())

    summary = profile.get("summary", "")
    if summary and len(summary) < 300:
        parts.append(summary)

    return " ".join(parts).strip()


# ═══════════════════════════════════════════════════════════════
#  PART 5: SIGNAL DETECTORS
#  Each one maps directly to a sentence in the JD.
# ═══════════════════════════════════════════════════════════════

def detect_deployment_evidence(history: list) -> tuple:
    """'Has shipped at least one end-to-end ... system to real users at
    meaningful scale.' Returns (score, has_scale_number)."""
    text = _career_full_text(history)
    hits = _count_phrase_hits(text, PRODUCTION_DEPLOYMENT_PHRASES)
    has_scale_number = bool(SCALE_NUMBER_PATTERN.search(text))
    return hits + (2 if has_scale_number else 0), has_scale_number

def rule_based_shipper_tilt(history: list) -> float:
    """Range roughly -1 (researcher language) to +1 (shipper language)."""
    text = _career_full_text(history)
    shipper_hits = _count_phrase_hits(text, SHIPPER_PHRASES)
    researcher_hits = _count_phrase_hits(text, RESEARCHER_PHRASES)
    total = shipper_hits + researcher_hits
    if total == 0:
        return 0.0
    return (shipper_hits - researcher_hits) / total

def detect_pure_research_profile(history: list) -> bool:
    """JD disqualifier: 'pure research environments ... without any
    production deployment — we will not move forward.' HARD disqualifier."""
    if not history:
        return False
    research_job_count = sum(
        1 for j in history
        if any(t in _lower(j.get("title", "")) for t in PURE_RESEARCH_TITLES)
    )
    deployment_score, _ = detect_deployment_evidence(history)
    return research_job_count == len(history) and research_job_count > 0 and deployment_score == 0

def detect_langchain_only_profile(history: list) -> bool:
    """JD disqualifier: recent (<12mo) 'AI experience' that's just LangChain
    + OpenAI calls, without pre-LLM-era production IR/ranking experience."""
    if not history:
        return False
    sorted_hist = sorted(history, key=lambda j: j.get("start_date") or "0000", reverse=True)
    recent_text = _job_text(sorted_hist[0])
    older_text = _career_full_text(sorted_hist[1:])

    recent_is_tutorial_style = (
        any(t in recent_text for t in LANGCHAIN_TUTORIAL_TERMS)
        and not any(t in recent_text for t in NLP_IR_TERMS)
    )
    has_pre_llm_ir_experience = any(t in older_text for t in PRE_LLM_IR_TERMS)
    return recent_is_tutorial_style and not has_pre_llm_ir_experience

def detect_architect_drift_no_code(history: list) -> bool:
    """JD disqualifier: senior people who haven't written production code
    in 18+ months because they moved into architecture/tech-lead roles."""
    current_jobs = [j for j in history if j.get("is_current")]
    if not current_jobs:
        return False
    current = current_jobs[0]
    is_architect_title = any(t in _lower(current.get("title", "")) for t in ARCHITECT_DRIFT_TITLES)
    duration = current.get("duration_months", 0) or 0
    return is_architect_title and duration >= 18

def detect_consulting_only(history: list) -> bool:
    """JD: people who have ONLY worked at consulting firms their entire
    career. Currently-at-a-consulting-firm-with-prior-product-experience
    is explicitly fine, so this only fires if EVERY job is a consulting firm."""
    if not history:
        return False
    def is_consulting(company):
        c = _lower(company)
        return any(firm in c for firm in CONSULTING_FIRMS)
    return all(is_consulting(j.get("company", "")) for j in history)

def detect_domain_mismatch(history: list, skills: list) -> bool:
    """JD: CV/speech/robotics expertise without significant NLP/IR exposure."""
    text = _career_full_text(history) + " " + " ".join(_lower(s.get("name", "")) for s in skills)
    has_cv_speech_robo = any(t in text for t in CV_SPEECH_ROBOTICS_TERMS)
    has_nlp_ir = any(t in text for t in NLP_IR_TERMS)
    return has_cv_speech_robo and not has_nlp_ir

def detect_title_chasing(history: list) -> bool:
    """JD: 'Senior -> Staff -> Principal' by switching companies every
    ~1.5 years. Average tenure under 18mo + monotonically increasing
    seniority across 3+ roles."""
    if len(history) < 3:
        return False
    sorted_hist = sorted(history, key=lambda j: j.get("start_date") or "0000")
    durations = [j.get("duration_months", 0) or 0 for j in sorted_hist]
    avg_tenure = sum(durations) / len(durations) if durations else 0
    ranks = [_seniority_rank(j.get("title", "")) for j in sorted_hist]
    increasing = all(
        r2 >= r1 for r1, r2 in zip(ranks, ranks[1:]) if r1 != -1 and r2 != -1
    )
    return avg_tenure < 18 and increasing


# ═══════════════════════════════════════════════════════════════
#  PART 6: AVAILABILITY (behavioral down-weighting)
#  JD: "a perfect-on-paper candidate who hasn't logged in for 6 months
#  and has a 5% recruiter response rate is ... not actually available."
# ═══════════════════════════════════════════════════════════════

def compute_availability_multiplier(last_active_val, response_rate_val) -> tuple:
    multiplier = 1.0
    notes = []

    if last_active_val:
        try:
            last_active_dt = datetime.fromisoformat(str(last_active_val).replace("Z", "+00:00"))
            days_inactive = (datetime.now(timezone.utc) - last_active_dt).days
            if days_inactive > INACTIVITY_DAYS_SEVERE:
                multiplier *= 0.5
                notes.append(f"inactive {days_inactive}d")
            elif days_inactive > INACTIVITY_DAYS_MILD:
                multiplier *= 0.8
                notes.append(f"inactive {days_inactive}d")
        except Exception:
            notes.append("last_active present but unparseable")
    else:
        notes.append("no activity data found")

    if response_rate_val is not None:
        try:
            rr = float(response_rate_val)
            if rr > 1:
                rr = rr / 100.0
            if rr < LOW_RESPONSE_RATE_THRESHOLD:
                multiplier *= 0.7
                notes.append(f"low recruiter response rate {rr:.0%}")
        except Exception:
            pass
    else:
        notes.append("no response-rate data found")

    return multiplier, "; ".join(notes)


# ═══════════════════════════════════════════════════════════════
#  PART 7: FINAL SCORE COMPOSITION
# ═══════════════════════════════════════════════════════════════

def compute_final_score(narrative_score, shipper_tilt, soft_flag_count, availability_multiplier) -> float:
    shipper_multiplier = 1 + SHIPPER_TILT_WEIGHT * max(-1.0, min(1.0, shipper_tilt))
    penalty_multiplier = max(MIN_PENALTY_MULTIPLIER, 1 - SOFT_FLAG_PENALTY * soft_flag_count)
    return narrative_score * shipper_multiplier * penalty_multiplier * availability_multiplier

def build_rationale(entry: dict) -> str:
    if entry["hard_disqualified"]:
        return f"HARD DISQUALIFIED: {entry['disqualify_reason']}"

    bits = [f"Narrative fit {entry['narrative_score']:.3f} vs JD."]

    gap = entry["keyword_stuffing_gap"]
    if gap is not None and gap > 0.05:
        bits.append(
            f"Caution: skills-list score is {gap:.3f} higher than narrative "
            "fit — possible keyword stuffing, weighted down accordingly."
        )

    if entry["shipper_tilt"] > 0.15:
        bits.append("Tilts shipper (production/iteration language dominant).")
    elif entry["shipper_tilt"] < -0.15:
        bits.append("Tilts researcher (publication/benchmark language dominant).")

    if entry["deployment_evidence_score"] >= 2:
        bits.append("Concrete deployment/scale evidence in career history.")

    flags_true = [k for k, v in entry["soft_flags"].items() if v]
    if flags_true:
        bits.append("Soft flags: " + ", ".join(flags_true) + ".")

    if entry["availability_multiplier"] < 1.0:
        bits.append(f"Down-weighted for availability ({entry['availability_notes']}).")

    return " ".join(bits)


# ═══════════════════════════════════════════════════════════════
#  PART 8: MAIN
# ═══════════════════════════════════════════════════════════════

def run():
    for required in [JD_EMBEDDING_PATH, CANDIDATE_EMBEDDINGS_PATH, CANDIDATES_WITH_TEXT_PATH]:
        if not Path(required).exists():
            print(f"\n[ERROR] Missing required input: {required}")
            print("  → Run embed_pipeline.py first.")
            return

    print(f"\n  Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    jd_vec = np.load(JD_EMBEDDING_PATH)

    npz = np.load(CANDIDATE_EMBEDDINGS_PATH, allow_pickle=True)
    emb_with_skills_all = npz["embeddings"]
    ids_from_npz = [str(i) for i in npz["ids"]]
    id_to_idx = {cid: idx for idx, cid in enumerate(ids_from_npz)}

    with open(CANDIDATES_WITH_TEXT_PATH, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"  Loaded {len(candidates):,} candidates.")

    print("\n  Building trap-resistant narrative texts (skills line excluded)...")
    narrative_texts = [build_narrative_text(c) for c in candidates]

    print("  Embedding persona anchors + narrative texts...")
    anchor_vecs = model.encode(
        [SHIPPER_ANCHOR_TEXT, RESEARCHER_ANCHOR_TEXT], normalize_embeddings=True
    )
    shipper_anchor_vec, researcher_anchor_vec = anchor_vecs[0], anchor_vecs[1]

    narrative_embs = []
    for i in range(0, len(narrative_texts), BATCH_SIZE):
        batch = narrative_texts[i : i + BATCH_SIZE]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        narrative_embs.append(vecs)
        print(f"\r  Encoded {min(i + BATCH_SIZE, len(narrative_texts))}/{len(narrative_texts)}", end="", flush=True)
    print()
    narrative_embs = np.vstack(narrative_embs)

    print("\n  Scoring candidates...")
    results = []
    for i, c in enumerate(candidates):
        cid = str(c.get("candidate_id", ids_from_npz[i] if i < len(ids_from_npz) else f"row_{i}"))
        history = c.get("career_history", []) or []
        skills = c.get("skills", []) or []
        profile = c.get("profile", {}) or {}

        narrative_score = float(narrative_embs[i] @ jd_vec)

        idx = id_to_idx.get(cid)
        keyword_score = float(emb_with_skills_all[idx] @ jd_vec) if idx is not None else None
        keyword_stuffing_gap = (keyword_score - narrative_score) if keyword_score is not None else None

        shipper_semantic_tilt = float(
            (narrative_embs[i] @ shipper_anchor_vec) - (narrative_embs[i] @ researcher_anchor_vec)
        )
        shipper_rule_tilt = rule_based_shipper_tilt(history)
        shipper_tilt = 0.6 * shipper_semantic_tilt + 0.4 * shipper_rule_tilt

        deployment_score, has_scale_evidence = detect_deployment_evidence(history)

        hard_flag_pure_research = detect_pure_research_profile(history)
        soft_flags = {
            "langchain_only_recent": detect_langchain_only_profile(history),
            "architect_drift_no_code": detect_architect_drift_no_code(history),
            "consulting_only": detect_consulting_only(history),
            "domain_mismatch_cv_speech_robotics": detect_domain_mismatch(history, skills),
            "title_chasing": detect_title_chasing(history),
        }
        soft_flag_count = sum(soft_flags.values())

        last_active_val = _find_first_present(c, LAST_ACTIVE_KEYS)
        response_rate_val = _find_first_present(c, RESPONSE_RATE_KEYS)
        availability_multiplier, availability_notes = compute_availability_multiplier(
            last_active_val, response_rate_val
        )

        final_score = compute_final_score(narrative_score, shipper_tilt, soft_flag_count, availability_multiplier)
        if hard_flag_pure_research:
            final_score = 0.0

        full_text = _career_full_text(history)
        lexical_signals = {
            "mentions_embedding_retrieval_terms": _count_phrase_hits(full_text, EMBEDDING_RETRIEVAL_TERMS) > 0,
            "mentions_vector_db_terms": _count_phrase_hits(full_text, VECTOR_DB_TERMS) > 0,
            "mentions_eval_framework_terms": _count_phrase_hits(full_text, EVAL_FRAMEWORK_TERMS) > 0,
        }

        entry = {
            "candidate_id": cid,
            "current_title": profile.get("current_title", ""),
            "current_company": profile.get("current_company", ""),
            "narrative_score": round(narrative_score, 4),
            "keyword_score": round(keyword_score, 4) if keyword_score is not None else None,
            "keyword_stuffing_gap": round(keyword_stuffing_gap, 4) if keyword_stuffing_gap is not None else None,
            "shipper_tilt": round(shipper_tilt, 4),
            "deployment_evidence_score": deployment_score,
            "has_scale_evidence": has_scale_evidence,
            "hard_disqualified": hard_flag_pure_research,
            "disqualify_reason": (
                "Entire career in pure-research roles with no production deployment evidence."
                if hard_flag_pure_research else ""
            ),
            "soft_flags": soft_flags,
            "soft_flag_count": soft_flag_count,
            "availability_multiplier": round(availability_multiplier, 3),
            "availability_notes": availability_notes,
            "lexical_signals": lexical_signals,
            "final_score": round(final_score, 4),
        }
        entry["rationale"] = build_rationale(entry)
        results.append(entry)

    # Eligible candidates first, then descending final_score within each group.
    results.sort(key=lambda r: (not r["hard_disqualified"], r["final_score"]), reverse=True)

    print("\n── Top 15 ranked candidates ──────────────────────")
    for rank, r in enumerate(results[:15], 1):
        print(f"\n  #{rank}  {r['candidate_id']}  score={r['final_score']:.4f}  "
              f"{r['current_title']} @ {r['current_company']}")
        print(f"       {r['rationale']}")

    n_disqualified = sum(1 for r in results if r["hard_disqualified"])
    n_stuffing_flagged = sum(
        1 for r in results if r["keyword_stuffing_gap"] is not None and r["keyword_stuffing_gap"] > 0.05
    )

    with open(OUTPUT_RANKED_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id", "current_title", "current_company", "final_score",
            "narrative_score", "keyword_score", "keyword_stuffing_gap",
            "shipper_tilt", "deployment_evidence_score", "hard_disqualified",
            "soft_flag_count", "availability_multiplier", "rationale",
        ])
        for r in results:
            writer.writerow([
                r["candidate_id"], r["current_title"], r["current_company"], r["final_score"],
                r["narrative_score"], r["keyword_score"], r["keyword_stuffing_gap"],
                r["shipper_tilt"], r["deployment_evidence_score"], r["hard_disqualified"],
                r["soft_flag_count"], r["availability_multiplier"], r["rationale"],
            ])

    print(f"""
╔══════════════════════════════════════════════════════╗
  SCORING COMPLETE
╠══════════════════════════════════════════════════════╣
  Candidates scored          : {len(results):,}
  Hard-disqualified           : {n_disqualified:,}
  Flagged for keyword stuffing: {n_stuffing_flagged:,}
  ──────────────────────────────────────────────────
  Output files:
    {OUTPUT_RANKED_JSON}
    {OUTPUT_SUMMARY_CSV}
╚══════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    run()