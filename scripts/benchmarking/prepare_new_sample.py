import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from utils import get_hash

ROOT = "Master_thesis/Phosphate-binding-site"
SEQRES = os.path.join(ROOT, "data/pdb_seqres.txt")
JSON_CACHE = os.path.join(ROOT, "cache/blast_cache_1e-16_0.5_0.95.json")
OUT_KEYS = os.path.join(ROOT, "data/sample_500_keys.txt")
OUT_FASTA = os.path.join(ROOT, "data/sample_500.fasta")

def main():
    print("Loading web cache...")
    with open(JSON_CACHE, 'r') as f:
        cache_data = json.load(f)
    
    all_cache_keys = list(cache_data.keys())
    print(f"Total available keys in cache: {len(all_cache_keys)}")
    
    # Sample 500 keys
    sample_keys_target = set(random.sample(all_cache_keys, min(500, len(all_cache_keys))))
    print(f"Targeted {len(sample_keys_target)} keys for the new sample.")

    print("Scanning pdb_seqres.txt for sequences...")
    found_keys = {} # hash_key -> sequence
    
    with open(SEQRES, 'r', encoding='utf-8', errors='ignore') as f:
        current_header = None
        for line in f:
            if line.startswith(">"):
                current_header = line.strip()
            else:
                seq = line.strip()
                if current_header and "mol:protein" in current_header:
                    pdb_id = current_header[1:5].upper()
                    seq_hash = get_hash(seq)
                    cache_key = f"{pdb_id}_{seq_hash}"
                    
                    if cache_key in sample_keys_target:
                        found_keys[cache_key] = seq
                        sample_keys_target.remove(cache_key)
                        if not sample_keys_target:
                            break
    
    print(f"Found {len(found_keys)} matching sequences.")
    
    with open(OUT_KEYS, 'w') as f_keys, open(OUT_FASTA, 'w') as f_fasta:
        for key, seq in found_keys.items():
            f_keys.write(f"{key}\n")
            f_fasta.write(f">{key}\n{seq}\n")
            
    print(f"Saved {len(found_keys)} keys to {OUT_KEYS}")
    print(f"Saved {len(found_keys)} sequences to {OUT_FASTA}")

if __name__ == "__main__":
    main()
