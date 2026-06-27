import gzip
import shutil

input_filename = "../data/candidates.jsonl"
output_filename = "candidat.jsonl.gz"

# 1. Compress the JSONL file to .gz
with open(input_filename, "rb") as f_in:
    with gzip.open(output_filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"Successfully compressed to {output_filename}")

# 2. Count the total number of candidates inside the .gz file
candidate_count = 0

with gzip.open(output_filename, "rt", encoding="utf-8") as f_gz:
    for line in f_gz:
        # Strip whitespace and ensure the line isn't empty
        if line.strip():
            candidate_count += 1

print(f"Total number of candidates in the final .gz file: {candidate_count}")