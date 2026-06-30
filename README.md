"# data_ai" 
echo "# data_ai" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Dhruvi-45/data_ai.git
git push -u origin main


!pip install groq
!pip install sentence-transformers faiss-gpu google-genai --quiet




git remote add origin https://github.com/Dhruvi-45/data_ai.git
git branch -M main
git push -u origin main

git status
git add .
git commit -m "debug till filtering"
git push


candidate-ranking/
│
├── README.md                    # Approach explanation + how to run
│
├── data/
│   ├── candidates.jsonl.gz      # Original compressed dataset
│   └── .gitkeep                 # (if data is too large, add to .gitignore)
│
├── precompute/
│   ├── 01_load_and_filter.py    # Load .gz, clean, dedup
│   ├── 02_embed_and_score.py    # Vector embeddings + hybrid scoring
│   ├── 03_llm_reasons.py        # Generate 1-line LLM reasons for top 100
│   └── run_precompute.sh        # Runs all 3 steps in order
│
├── outputs/                     # Precomputed results (committed to repo)
│   ├── filtered_candidates.jsonl
│   ├── scored_candidates.json
│   └── top100_with_reasons.json # Final output used at runtime
│
├── runtime/
│   └── serve_rankings.py        # Loads precomputed top100, serves in <5 min
│
├── notebooks/
│   └── exploration.ipynb        # Your Colab prototyping notebook
│
├── requirements.txt
└── .gitignore                   # Add: data/*.jsonl, __pycache__, .env