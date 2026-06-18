"""
embed_pipeline.py
==================
Stage 4 of the Redrob AI pipeline: Embedding generation.

WHAT THIS DOES:
───────────────
1. Extracts a clean, signal-only text representation from the JD
   (skills required, role context, ideal candidate description).
2. Extracts a clean text representation from each candidate
   (tech stack, career trajectory, strongest relevant job).
3. Embeds the JD once using BAAI/bge-small-en-v1.5.
4. Embeds all candidates in batches (low memory, CPU-safe).
5. Saves:
     - jd_embedding.npy          → JD embedding vector (1 x 384)
     - candidate_embeddings.npz  → all candidate vectors + ID index
     - candidates_with_text.json → candidates + their extracted text
                                    (useful for debugging / scoring)

HOW TO USE IN GOOGLE COLAB:
────────────────────────────
# Step 1: Install
!pip install sentence-transformers --quiet

# Step 2: Mount Drive if needed
from google.colab import drive
drive.mount('/content/drive')

# Step 3: Update INPUT_PATH below and run
!python embed_pipeline.py

RUNTIME ESTIMATE (Colab free CPU):
────────────────────────────────────
30,000 candidates × batch_size=64 ≈ 3–5 minutes
Model download on first run ≈ 1 minute (cached after)
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── CONFIGURE ────────────────────────────────────────────────────────────────
INPUT_PATH              = "kept_candidates.json"         # ← your filtered file
JD_EMBEDDING_PATH       = "jd_embedding.npy"             # ← output
CANDIDATE_EMBEDDINGS_PATH = "candidate_embeddings.npz"   # ← output
CANDIDATES_WITH_TEXT_PATH = "candidates_with_text.json"  # ← output (debug)

MODEL_NAME  = "BAAI/bge-small-en-v1.5"
BATCH_SIZE  = 64   # safe for Colab CPU RAM; increase to 128 if you have more RAM
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════
#  PART 1: JD TEXT EXTRACTION
#  We extract only the signal-rich parts of the JD.
#  The JD is hardcoded here since it doesn't change.
#  Disqualifiers, location, culture, salary are intentionally EXCLUDED.
# ═══════════════════════════════════════════════════════════════

JD_QUERY_TEXT = """
Senior AI Engineer role at a product company building candidate ranking,
retrieval, and matching systems for a talent intelligence platform.

Required skills and experience:
Production experience with embedding-based retrieval systems using
sentence-transformers, BGE, E5, OpenAI embeddings, or similar models,
including handling embedding drift, index refresh, and retrieval quality
regression in production environments.
Production experience with vector databases or hybrid search infrastructure
such as Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch, FAISS, or OpenSearch.
Strong Python skills with focus on code quality.
Hands-on experience designing evaluation frameworks for ranking systems
including NDCG, MRR, MAP, offline benchmarks, A/B testing, and recruiter
feedback loops.

Ideal background:
Six to eight years total experience with four to five years in applied ML
or AI roles at product companies, not pure services or consulting.
Has shipped at least one end-to-end ranking, search, or recommendation
system to real users at meaningful scale.
Experience with NLP, information retrieval, and ranking before the LLM era.
Strong opinions on hybrid versus dense retrieval, offline versus online
evaluation, and when to fine-tune versus prompt LLMs.

Nice to have:
LLM fine-tuning with LoRA, QLoRA, or PEFT.
Learning-to-rank models using XGBoost or neural approaches.
Experience in HR-tech, recruiting, or marketplace products.
Open-source contributions in AI or ML.
""".strip()


# ═══════════════════════════════════════════════════════════════
#  PART 2: CANDIDATE TEXT EXTRACTION
#  We build a short, dense, signal-only text from each candidate.
#  This is what gets embedded and compared to the JD.
#  Structure: role context → tech skills → career trajectory → top job
# ═══════════════════════════════════════════════════════════════

# Skills that are directly relevant to this JD
# Used to surface relevant skills from the candidate's skill list
RELEVANT_SKILL_KEYWORDS = {
    # Core retrieval / search
    "elasticsearch", "opensearch", "solr", "lucene", "bm25",
    "faiss", "annoy", "hnsw", "qdrant", "pinecone", "weaviate",
    "milvus", "vespa", "vector search", "hybrid search",
    # Embeddings / NLP
    "sentence-transformers", "sentence transformers", "bge", "e5",
    "bert", "roberta", "transformers", "hugging face", "huggingface",
    "word2vec", "glove", "fasttext", "spacy", "nltk",
    "nlp", "natural language processing", "information retrieval",
    # Ranking / RecSys
    "learning to rank", "ltr", "ranknet", "lambdamart",
    "xgboost", "lightgbm", "gradient boosting",
    "recommendation system", "recsys", "collaborative filtering",
    "retrieval augmented generation", "rag",
    # LLMs / fine-tuning
    "llm", "large language model", "fine-tuning", "finetuning",
    "lora", "qlora", "peft", "rlhf", "instruction tuning",
    "openai", "gpt", "claude", "gemini", "llama", "mistral",
    # Eval
    "ndcg", "mrr", "map", "a/b testing", "a/b test",
    "evaluation framework", "offline evaluation", "online evaluation",
    # ML infra
    "mlops", "model serving", "triton", "torchserve", "bentoml",
    "kubeflow", "airflow", "spark", "kafka",
    # Core languages / tools
    "python", "pytorch", "tensorflow", "jax", "scikit-learn", "sklearn",
}

def _lower(text) -> str:
    return text.lower().strip() if text else ""

def _extract_relevant_skills(skills: list) -> list:
    """Return only skills that are relevant to this JD."""
    relevant = []
    for skill in skills:
        name = skill.get("name", "")
        if any(kw in _lower(name) for kw in RELEVANT_SKILL_KEYWORDS):
            relevant.append(name)
    return relevant

def _get_top_job(career_history: list) -> dict | None:
    """
    Find the most relevant job to this JD.
    Strategy: score each job by how many relevant keywords appear
    in its title + description, return the highest-scoring one.
    """
    if not career_history:
        return None

    best_job   = None
    best_score = -1

    for job in career_history:
        text = _lower(
            job.get("title", "") + " " + job.get("description", "")
        )
        score = sum(1 for kw in RELEVANT_SKILL_KEYWORDS if kw in text)
        # Slight boost for current roles (recency matters)
        if job.get("is_current"):
            score += 2
        if score > best_score:
            best_score = score
            best_job   = job

    return best_job

def _build_candidate_text(candidate: dict) -> str:
    """
    Build a clean, embedding-ready text for a candidate.
    Deliberately short and dense — no fluff, just signal.

    Structure:
      [1] Current role + years of experience
      [2] Relevant technical skills
      [3] Career trajectory (most recent 3 roles)
      [4] Top relevant job description (truncated)
      [5] Education (field only)
    """
    profile  = candidate.get("profile", {})
    history  = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])
    education = candidate.get("education", [])

    parts = []

    # [1] Role context
    title   = profile.get("current_title", "")
    yoe     = profile.get("years_of_experience", "")
    company = profile.get("current_company", "")
    headline = profile.get("headline", "")

    if title:
        role_line = f"{title}"
        if company:
            role_line += f" at {company}"
        if yoe:
            role_line += f" with {yoe} years of experience"
        parts.append(role_line + ".")
    if headline and headline != title:
        parts.append(headline + ".")

    # [2] Relevant skills
    relevant_skills = _extract_relevant_skills(skills)
    if relevant_skills:
        parts.append(
            "Technical skills: " + ", ".join(relevant_skills[:20]) + "."
        )
    else:
        # Fall back to all skills if none match (avoids empty embedding)
        all_skills = [s.get("name", "") for s in skills[:15]]
        if all_skills:
            parts.append("Skills: " + ", ".join(all_skills) + ".")

    # [3] Career trajectory — last 3 roles
    # Sort by start_date descending (most recent first)
    sorted_history = sorted(
        history,
        key=lambda j: j.get("start_date") or "0000",
        reverse=True
    )
    trajectory_parts = []
    for job in sorted_history[:3]:
        t = job.get("title", "")
        c = job.get("company", "")
        d = job.get("duration_months", 0) or 0
        years = round(d / 12, 1)
        if t and c:
            trajectory_parts.append(f"{t} at {c} ({years}y)")
    if trajectory_parts:
        parts.append("Career: " + " → ".join(trajectory_parts) + ".")

    # [4] Most relevant job description (up to 200 chars — enough signal, not noise)
    top_job = _get_top_job(history)
    if top_job:
        desc = top_job.get("description", "")
        if desc:
            # Truncate cleanly at a word boundary
            if len(desc) > 200:
                desc = desc[:200].rsplit(" ", 1)[0] + "..."
            parts.append(f"Key work: {desc}")

    # [5] Education — field only (institution tier is handled in scoring, not embedding)
    for edu in education[:1]:
        field = edu.get("field_of_study", "")
        degree = edu.get("degree", "")
        if field:
            parts.append(f"Education: {degree} in {field}.".strip())

    # [6] Summary if present and short
    summary = profile.get("summary", "")
    if summary and len(summary) < 300:
        parts.append(summary)

    return " ".join(parts).strip()


# ═══════════════════════════════════════════════════════════════
#  PART 3: EMBEDDING GENERATION
# ═══════════════════════════════════════════════════════════════

def embed_jd(model: SentenceTransformer) -> np.ndarray:
    """
    Embed the JD query text.
    BGE models need the query prefix "Represent this sentence: " for queries.
    The candidate texts are "documents" — no prefix needed for those.
    """
    print("  Embedding JD...")
    # BGE asymmetric retrieval: queries get this prefix, documents do not
    query = "Represent this sentence: " + JD_QUERY_TEXT
    vec = model.encode(query, normalize_embeddings=True)
    print(f"  JD embedding shape: {vec.shape}")
    return vec

def embed_candidates(
    candidates: list,
    model: SentenceTransformer,
    batch_size: int = 64
) -> tuple[np.ndarray, list, list]:
    """
    Embed all candidates in batches.

    Returns:
        embeddings  : np.ndarray of shape (N, 384)
        ids         : list of candidate_id strings (same order as embeddings)
        texts       : list of extracted text strings (for debugging)
    """
    texts = []
    ids   = []

    print(f"  Extracting text from {len(candidates):,} candidates...")
    for c in candidates:
        ids.append(c.get("candidate_id", "UNKNOWN"))
        texts.append(_build_candidate_text(c))

    print(f"  Encoding in batches of {batch_size}...")
    total_batches = (len(texts) + batch_size - 1) // batch_size

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch     = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        vecs      = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_embeddings.append(vecs)

        # Progress
        pct = batch_num / total_batches * 100
        done = int(pct / 2)
        bar  = "█" * done + "░" * (50 - done)
        print(f"\r  [{bar}] {pct:5.1f}%  batch {batch_num}/{total_batches}", end="", flush=True)

    print()  # newline after progress
    embeddings = np.vstack(all_embeddings)
    print(f"  Candidate embeddings shape: {embeddings.shape}")
    return embeddings, ids, texts


# ═══════════════════════════════════════════════════════════════
#  PART 4: SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════

def save_outputs(
    jd_vec: np.ndarray,
    cand_vecs: np.ndarray,
    ids: list,
    candidates: list,
    texts: list
):
    # 1. JD embedding
    np.save(JD_EMBEDDING_PATH, jd_vec)
    print(f"  Saved: {JD_EMBEDDING_PATH}")

    # 2. Candidate embeddings + ID index (compressed, much smaller than .npy)
    np.savez_compressed(
        CANDIDATE_EMBEDDINGS_PATH,
        embeddings=cand_vecs,
        ids=np.array(ids)
    )
    print(f"  Saved: {CANDIDATE_EMBEDDINGS_PATH}.npz")

    # 3. Candidates with extracted text (for scoring stage + debugging)
    candidates_with_text = []
    for c, text in zip(candidates, texts):
        entry = dict(c)
        entry["_embedding_text"] = text   # prefixed with _ so it's clearly added
        candidates_with_text.append(entry)

    with open(CANDIDATES_WITH_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, record in enumerate(candidates_with_text):
            comma = "," if i < len(candidates_with_text) - 1 else ""
            f.write("  " + json.dumps(record, ensure_ascii=False) + comma + "\n")
        f.write("]\n")
    print(f"  Saved: {CANDIDATES_WITH_TEXT_PATH}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def run():
    input_path = Path(INPUT_PATH)
    if not input_path.exists():
        print(f"\n[ERROR] Input file not found: {input_path}")
        print("  → Update INPUT_PATH at the top of this script.")
        return

    # Load candidates
    print(f"\n  Loading candidates from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"  Loaded {len(candidates):,} candidates.")

    # Load model (downloads ~130MB on first run, cached after)
    print(f"\n  Loading model: {MODEL_NAME}")
    print("  (Downloads ~130MB on first run, cached after)")
    model = SentenceTransformer(MODEL_NAME)
    print("  Model loaded.")

    # Embed JD
    print("\n── JD Embedding ─────────────────────────────────")
    jd_vec = embed_jd(model)

    # Embed candidates
    print("\n── Candidate Embeddings ─────────────────────────")
    cand_vecs, ids, texts = embed_candidates(candidates, model, batch_size=BATCH_SIZE)

    # Quick sanity check: print top-5 most similar to JD
    print("\n── Sanity check: top-5 most similar to JD ───────")
    sims = cand_vecs @ jd_vec   # cosine sim (already normalised)
    top5_idx = np.argsort(sims)[::-1][:5]
    for rank, idx in enumerate(top5_idx, 1):
        c   = candidates[idx]
        sim = sims[idx]
        title   = c["profile"].get("current_title", "?")
        company = c["profile"].get("current_company", "?")
        print(f"  #{rank}  sim={sim:.4f}  {title} @ {company}")
        print(f"       text: {texts[idx][:120]}...")

    # Save
    print("\n── Saving outputs ───────────────────────────────")
    save_outputs(jd_vec, cand_vecs, ids, candidates, texts)

    print(f"""
╔══════════════════════════════════════════════════════╗
  EMBEDDING COMPLETE
╠══════════════════════════════════════════════════════╣
  Candidates embedded : {len(candidates):,}
  Embedding dimension : {cand_vecs.shape[1]}
  ──────────────────────────────────────────────────
  Output files:
    {JD_EMBEDDING_PATH}
    {CANDIDATE_EMBEDDINGS_PATH}.npz
    {CANDIDATES_WITH_TEXT_PATH}
  ──────────────────────────────────────────────────
  Next step → run score_candidates.py
╚══════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    run()