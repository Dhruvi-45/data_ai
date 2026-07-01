

To execute the entire end-to-end pipeline and generate the final submission file from the root directory, run the following single command:

```bash
bash run_all.sh ./output/candidates.jsonl.gz ./submission.csv

# Core Frameworks & Utilities
python-dotenv
streamlit
pandas
numpy

# Deep Learning & Embeddings
torch --index-url https://download.pytorch.org/whl/cu121
sentence-transformers

# Vector Search (GPU accelerated)
faiss-gpu

# AI Client SDKs
groq
google-genai





├── data/
│   ├── candidates.jsonl
│   └── candidates.jsonl.gz         # Original 100k dataset
├── precompute/
│   ├── check_gpu.py
│   ├── count_gz.py
│   ├── count_jn.py
│   ├── embed_search.py             # Generates .npz, .npy, .json artifacts
│   ├── filter_candidates_gz.py     # Stage 1 filter (GZ -> kept_candidates)
│   ├── honeypot_filter.py          # Stage 2 filter (kept -> honey_pot_candidates.json)
│   └── jntogz.py
├── runtime/
│   ├── __init__.py
│   └── rank.py                     # Final ranker script
├── output/                         # (Or your designated output directory)
├── .gitattributes                  # Git LFS tracking file
├── requirements.txt                # Dependencies list
├── run_all.sh                      # End-to-end pipeline script
├── submission_metadata.yaml        # Required hackathon metadata
└── README.md                       # This file