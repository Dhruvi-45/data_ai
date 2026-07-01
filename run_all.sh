#!/bin/bash
# Exit immediately if any command fails
set -e

# $1 captures the input path (e.g., ./output/candidates.jsonl.gz)
# $2 captures the output path (e.g., ./submission.csv)

echo "Stage 1: Filtering initial dataset from: $1"
python precompute/filter_candidates_gz.py --input "$1"

echo "Stage 2: Running honeypot filter..."
python precompute/honeypot_filter.py

echo "Stage 3: Generating embeddings..."
python precompute/embed_search.py

echo "Stage 4: Running final ranker..."
python runtime/rank.py --candidates "$1" --out "$2"

echo "Success! Final submission file generated at $2"