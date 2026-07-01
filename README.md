# 1. Uninstall the current CPU-only version
pip uninstall torch torchvision torchaudio -y

# 2. Install the CUDA-enabled version (Adjust cu121/cu124 based on your CUDA version if needed, but 12.1/12.4 are standard)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


### Code Reproduction

To execute the entire end-to-end pipeline and generate the final submission file from the root directory, run the following single command:

```bash
bash run_all.sh ./output/candidates.jsonl.gz ./submission.csv






"# data_ai" 
echo "# data_ai" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Dhruvi-45/data_ai.git
git push -u origin main

python -m pip install faiss-gpu
!pip install groq
!pip install sentence-transformers faiss-gpu google-genai --quiet




git remote add origin https://github.com/Dhruvi-45/data_ai.git
git branch -M main
git push -u origin main

git status
git add .
git commit -m "streamlit"
git push






pip install sentence-transformers faiss-gpu google-genai --quiet
pip install groq
py -3.10 -m pip install sentence-transformers faiss-cpu groq
py -3.10 -m pip install sentence-transformers faiss-gpu groq
py -3.10 -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall




py -3.10 -m pip install python-dotenv
streamlit
pandas
numpy






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