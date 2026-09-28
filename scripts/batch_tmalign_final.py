
import json
import os
import sys
import time

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from tmalign_wrapper import run_tmalign
from utils import to_wsl_path

def main():
    root = PROJECT_ROOT
    cache_path = os.path.join(root, "cache", "blast_cache_1e-16_0.5.json")
    tmalign_exec_win = os.path.join(root, "scripts", "Shell scripts", "TMalign")
    tmalign_exec = to_wsl_path(tmalign_exec_win)
    ref_dir = os.path.join(root, "data", "PDB structures", "reference structures")
    hits_dir = os.path.join(root, "data", "PDB structures", "blast hits")
    output_path = os.path.join(root, "results", "tmalign_results.json")

    # Load existing results if they exist (for resuming)
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            try:
                results = json.load(f)
                print(f"Resuming from {len(results)} existing entries in {output_path}")
            except:
                results = {}
    else:
        results = {}

    with open(cache_path, 'r') as f:
        cache = json.load(f)

    count = 0
    total = len(cache)
    start_time = time.time()

    for uniprot_id, data in cache.items():
        if uniprot_id in results:
            continue

        ref_pdb = data.get("initiating_pdb")
        hits = data.get("hits", [])
        
        if not ref_pdb or not hits:
            results[uniprot_id] = {"reference": ref_pdb, "alignments": [], "status": "no_hits"}
            continue

        # Find ref file
        ref_path = None
        for ext in [".pdb", ".cif"]:
            p = os.path.join(ref_dir, f"{ref_pdb}{ext}")
            if os.path.exists(p):
                ref_path = p
                break
        
        if not ref_path:
            results[uniprot_id] = {"reference": ref_pdb, "alignments": [], "status": "ref_not_found"}
            continue

        results[uniprot_id] = {
            "reference": ref_pdb,
            "alignments": []
        }

        for hit_entry in hits:
            if isinstance(hit_entry, dict):
                hit_pdb = hit_entry.get("pdb_id")
            else:
                hit_pdb = hit_entry

            if not hit_pdb: continue

            # Find hit file
            hit_path = None
            for ext in [".pdb", ".cif"]:
                p = os.path.join(hits_dir, f"{hit_pdb}{ext}")
                if os.path.exists(p):
                    hit_path = p
                    break
            
            if not hit_path: continue

            # ALIGN HIT TO REFERENCE (Moving hit -> reference space)
            res = run_tmalign(hit_path, ref_path, tmalign_executable=tmalign_exec)
            if res:
                res.pop("raw_output", None)
                res["hit"] = hit_pdb
                results[uniprot_id]["alignments"].append(res)

        count += 1
        if count % 50 == 0:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            elapsed = time.time() - start_time
            avg_time = elapsed / count
            remaining = (total - len(results)) * avg_time
            print(f"Progress: {len(results)}/{total} ({len(results)/total*100:.1f}%) - Est. remaining: {remaining/3600:.1f}h")

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted! Results saved to {output_path}")

if __name__ == "__main__":
    main()
