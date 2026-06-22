import gzip

# Check the COMPRESSED .jsonl.gz file directly
count = 0
with gzip.open('candidates.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            count += 1
print(f"Original .jsonl lines: {count}")