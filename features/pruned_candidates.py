import json

def prune_candidate_data(input_path, output_path):
    # 1. Load the data from your sample/filtered json file
    with open(input_path, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    
    print(f"Loaded {len(candidates)} candidates. Cleaning columns...")

    # 2. Iterate through each candidate object and prune the unwanted keys
    for candidate in candidates:
        # Remove 'anonymized_name' from the 'profile' object
        if "profile" in candidate and "anonymized_name" in candidate["profile"]:
            del candidate["profile"]["anonymized_name"]
            
        # Remove 'grade' and 'tier' from every entry in the 'education' array
        if "education" in candidate and isinstance(candidate["education"], list):
            for edu in candidate["education"]:
                if "grade" in edu:
                    del edu["grade"]
                if "tier" in edu:
                    del edu["tier"]
                    
        # Remove top-level 'certifications' column
        if "certifications" in candidate:
            del candidate["certifications"]
            
        # Remove top-level 'languages' column
        if "languages" in candidate:
            del candidate["languages"]

    # 3. Save the pruned dataset to a new file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Cleaned data successfully saved to '{output_path}'")

# --- Run the Function ---
if __name__ == "__main__":
    # Change these filenames to match your local setup
    input_file = "cleaned_candidates.json"
    output_file = "pruned_candidates.json"
    
    prune_candidate_data(input_file, output_file)