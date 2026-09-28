
import json
import os
import subprocess
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Configuration - using WSL paths
ROOT = "/mnt/d/AI/Master_thesis/Phosphate-binding-site"
CACHE_FILE = "blast_cache_1e-16_0.5_0.95.json"
CACHE_PATH = f"{ROOT}/cache/{CACHE_FILE}"
TMALIGN_EXEC = f"{ROOT}/scripts/Shell scripts/TMalign"
REF_DIR = f"{ROOT}/data/PDB structures/reference structures"
HITS_DIR = f"{ROOT}/data/PDB structures/blast hits"
OUTPUT_PATH = f"{ROOT}/results/tmalign_results_{CACHE_FILE.replace('.json', '')}.json"

def run_tmalign_linux(hit_path, ref_path):
    """Linux-native TMalign runner (no WSL overhead, no sleep)"""
    matrix_file = f"/tmp/matrix_{os.getpid()}_{time.time_ns()}.txt"
    cmd = [TMALIGN_EXEC, hit_path, ref_path, "-m", matrix_file]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse TM-score and RMSD
        rmsd_match = re.search(r"RMSD=\s+([\d.]+)", output)
        tm_score_1_match = re.search(r"TM-score=\s+([\d.]+)\s+\(if normalized by length of Chain_1", output)
        tm_score_2_match = re.search(r"TM-score=\s+([\d.]+)\s+\(if normalized by length of Chain_2", output)
        
        transformation = None
        if os.path.exists(matrix_file):
            with open(matrix_file, 'r') as f:
                content = f.read()
            m_lines = re.findall(r"^\s*\d\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)", content, re.MULTILINE)
            if len(m_lines) == 3:
                t = [float(line[0]) for line in m_lines]
                u = [[float(line[1]), float(line[2]), float(line[3])] for line in m_lines]
                transformation = {"t": t, "u": u}
            os.remove(matrix_file)

        return {
            "rmsd": float(rmsd_match.group(1)) if rmsd_match else None,
            "tm_score_1": float(tm_score_1_match.group(1)) if tm_score_1_match else None,
            "tm_score_2": float(tm_score_2_match.group(1)) if tm_score_2_match else None,
            "transformation": transformation
        }
    except Exception:
        if os.path.exists(matrix_file): os.remove(matrix_file)
        return None

def process_uniprot(uniprot_id, data, available_refs, available_hits):
    ref_pdb = data.get("initiating_pdb")
    hits = data.get("hits", [])
    if not ref_pdb or not hits:
        return uniprot_id, {"reference": ref_pdb, "alignments": [], "status": "no_hits"}

    ref_file = available_refs.get(ref_pdb)
    if not ref_file:
        return uniprot_id, {"reference": ref_pdb, "alignments": [], "status": "ref_not_found"}

    alignments = []
    for hit_entry in hits:
        hit_pdb = hit_entry.get("pdb_id") if isinstance(hit_entry, dict) else hit_entry
        hit_file = available_hits.get(hit_pdb)
        if not hit_file: continue

        res = run_tmalign_linux(hit_file, ref_file)
        if res:
            res["hit"] = hit_pdb
            alignments.append(res)
    
    return uniprot_id, {"reference": ref_pdb, "alignments": alignments}

def main():
    print("Loading cache and indexing files...")
    with open(CACHE_PATH, 'r') as f:
        cache = json.load(f)

    # Pre-index files for speed
    def index_dir(d):
        idx = {}
        for f in os.listdir(d):
            if f.endswith((".pdb", ".cif")):
                idx[f[:4]] = os.path.join(d, f)
        return idx

    available_refs = index_dir(REF_DIR)
    available_hits = index_dir(HITS_DIR)

    results = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r') as f:
            try: results = json.load(f)
            except: pass
    
    initial_done_count = len(results)
    
    # Pre-filter to process ONLY entries with hits and existing reference files
    to_process = {}
    for k, v in cache.items():
        if k in results:
            continue
        
        ref_pdb = v.get("initiating_pdb")
        hits = v.get("hits", [])
        
        if hits and ref_pdb in available_refs:
            to_process[k] = v
        else:
            # Mark these as skipped/no-op in results so we don't check them again
            results[k] = {
                "reference": ref_pdb, 
                "alignments": [], 
                "status": "skipped_no_hits_or_ref"
            }

    print(f"Total Cache: {len(cache)} | Already Done (from previous run): {initial_done_count} | Skipped (No hits/ref): {len(results) - initial_done_count} | To Process (Valid): {len(to_process)}")

    num_workers = multiprocessing.cpu_count()
    print(f"Starting parallel execution with {num_workers} workers...")

    start_time = time.time()
    count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_uniprot, uid, data, available_refs, available_hits): uid 
                   for uid, data in to_process.items()}
        
        for future in as_completed(futures):
            uid, result = future.result()
            results[uid] = result
            count += 1
            
            if count % 100 == 0:
                elapsed = time.time() - start_time
                avg = elapsed / count
                remaining = (len(to_process) - count) * avg
                print(f"Progress: {len(results)}/{len(cache)} ({len(results)/len(cache)*100:.1f}%) - Est. remaining: {remaining/3600:.2f}h")
                # Save checkpoint
                with open(OUTPUT_PATH, 'w') as f:
                    json.dump(results, f, indent=2)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()
