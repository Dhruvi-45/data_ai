import gzip
import json

candidates = []

with gzip.open('candidates.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:          # ← skip blank lines, don't break!
            continue          # ← common bug: people use 'break' here
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # skip malformed lines

print(f"Loaded: {len(candidates)}")  # Should be ~100k




# Check the ORIGINAL .jsonl file directly
count = 0
with open('candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            count += 1
print(f"Original .jsonl lines: {count}")