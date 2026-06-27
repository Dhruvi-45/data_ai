# Check the ORIGINAL .jsonl file directly
count = 0
with open('../data/candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            count += 1
print(f"Original .jsonl lines: {count}")