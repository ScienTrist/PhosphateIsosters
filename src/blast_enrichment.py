import collections
import json
import os
import io
import time
import hashlib
from mmseqs_handler import MMseqsHandler
from rcsbapi.data import DataQuery
from rcsb_search import filter_by_quality

# Define base cache directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

SEQUENCE_CACHE_FILE = os.path.join(CACHE_DIR, "pdb_sequence_cache.json")
ALL_PROTEINS_FASTA = os.path.join(DATA_DIR, "pdb_proteins_all.fasta")

def get_cache_path(evalue, identity, res, rfree, method):
    """Generates a unique cache path based on a descriptive string of search parameters."""
    import math
    q_short = "Xray" if method == "X-RAY DIFFRACTION" else "All"
    try:
        e_exponent = int(abs(math.log10(evalue)))
    except:
        e_exponent = "var"
        
    config_id = f"{e_exponent}_{identity}_{res}_{rfree}_{q_short}"
    cache_path = os.path.join(CACHE_DIR, f"blast_cache_{config_id}.json")
    return cache_path

def load_json_cache(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return {}
    return {}

def save_json_cache(cache, path):
    with open(path, 'w') as f:
        json.dump(cache, f, indent=2)

def get_sequences_from_fasta(pdb_ids, fasta_path):
    """
    Extracts ALL sequences for a specific set of PDB IDs.
    Handles headers like '>3cip_A' or '>4HHB_A'.
    Returns a dict mapping full_header_id -> sequence.
    """
    pdb_ids_upper = set(pid.upper() for pid in pdb_ids)
    found_sequences = {}
    
    if not os.path.exists(fasta_path):
        print(f"Error: {fasta_path} not found.")
        return found_sequences

    print(f"  Scanning local FASTA for sequences belonging to {len(pdb_ids_upper)} PDBs...")
    
    with open(fasta_path, 'r') as f:
        current_full_id = None
        current_pdb_id = None
        current_seq = []
        
        for line in f:
            if line.startswith(">"):
                # Process the record we just finished reading
                if current_full_id and current_pdb_id in pdb_ids_upper:
                    found_sequences[current_full_id] = "".join(current_seq)
                
                # Parse new header: >4HHB_A ...
                # We take the first word as the full ID
                header_parts = line[1:].split()
                if header_parts:
                    current_full_id = header_parts[0].upper()
                    current_pdb_id = current_full_id[:4]
                else:
                    current_full_id = None
                    current_pdb_id = None
                current_seq = []
            else:
                current_seq.append(line.strip())
        
        # Final entry in file
        if current_full_id and current_pdb_id in pdb_ids_upper:
            found_sequences[current_full_id] = "".join(current_seq)
            
    print(f"  Extraction complete: Found {len(found_sequences)} sequences across these PDBs.")
    return found_sequences

def run_homology_enrichment(grouped_data, evalue=1e-16, identity=0.5,
                           similarity_threshold=0.95, min_res=2.7, max_r_free=0.25, 
                           method="X-RAY DIFFRACTION", run_id=None, 
                           sensitivity=7.5, search_cov=0.0, db_cluster_cov=0.8, 
                           downloader=None, **kwargs):
    """Main entry point for finding homologs."""
    handler = MMseqsHandler(PROJECT_ROOT)
    tag = run_id if run_id else f"{int(time.time())}"
    
    # 1. Identify PDB IDs in your SUBSET
    all_query_pdbs = set()
    for major in grouped_data:
        for sub in grouped_data[major]:
            for uid_str, data in grouped_data[major][sub].items():
                for pdb in data["pdbs"]:
                    all_query_pdbs.add(pdb.upper())
    
    # 2. Get sequences LOCALLY (All chains/entities)
    pdb_to_seq = get_sequences_from_fasta(all_query_pdbs, ALL_PROTEINS_FASTA)
    
    if not pdb_to_seq:
        print("Error: No sequences found for subset. Cannot proceed.")
        return {}

    subset_fasta = os.path.join(CACHE_DIR, f"subset_queries_{tag}.fasta")
    with open(subset_fasta, 'w') as f:
        for full_id, seq in pdb_to_seq.items():
            f.write(f">{full_id}\n{seq}\n")

    # 3. Cluster the SUBSET
    print(f"Clustering subset sequences ({len(pdb_to_seq)} chains from {len(all_query_pdbs)} PDBs)...")
    subset_cluster_prefix = os.path.join(CACHE_DIR, f"subset_clustered_{tag}")
    if not handler.cluster_database(subset_fasta, subset_cluster_prefix, min_seq_id=similarity_threshold, coverage=db_cluster_cov):
        print("Error: MMseqs2 clustering failed.")
        return {}
    
    subset_rep_fasta = subset_cluster_prefix + "_rep_seq.fasta"
    subset_cluster_tsv = subset_cluster_prefix + "_cluster.tsv"
    subset_cluster_map = handler.load_cluster_map(subset_cluster_tsv)

    # EARLY DOWNLOAD: Start downloading seeds while searching
    if downloader:
        # Extract unique PDB IDs from cluster representatives
        seeds = set(rep_id[:4].upper() for rep_id in subset_cluster_map.keys())
        print(f"\n  [Early Download] Identified {len(seeds)} seed structures. Starting background download...")
        downloader.start_background_download(list(seeds), category="references")

    # 4. Search
    search_results_tsv = os.path.join(CACHE_DIR, f"mmseqs_search_hits_{tag}.tsv")
    print(f"Searching against full PDB (Sens: {sensitivity}, Cov: {search_cov})...")
    if not handler.search(subset_rep_fasta, ALL_PROTEINS_FASTA, search_results_tsv, 
                          min_seq_id=identity, evalue=evalue, coverage=search_cov, sensitivity=sensitivity):
        print("Error: MMseqs2 search failed.")
        return {}
    
    raw_hits = handler.parse_search_results(search_results_tsv)

    # 5. Filter and Expand
    original_pdb_set = set(all_query_pdbs)
    unique_hit_pdbs = set()
    for rep_id, hits in raw_hits.items():
        for hit in hits:
            if hit["pdb_id"].upper() not in original_pdb_set:
                unique_hit_pdbs.add(hit["pdb_id"].upper())

    print(f"Filtering {len(unique_hit_pdbs)} candidate homologs by quality...")
    high_quality_hits = set(filter_by_quality(list(unique_hit_pdbs), min_res=min_res, max_r_free=max_r_free, method=method))

    all_homologs = {}
    for rep_id_full, hits in raw_hits.items():
        rep_pdb = rep_id_full[:4].upper()
        
        # Deduplicate hits by PDB ID (keeping the best chain hit for each)
        best_hits_per_pdb = {}
        for h in hits:
            pid = h["pdb_id"].upper()
            if pid in high_quality_hits:
                if pid not in best_hits_per_pdb:
                    best_hits_per_pdb[pid] = h
                else:
                    # Update if current hit is better (lower evalue or higher identity)
                    existing = best_hits_per_pdb[pid]
                    if (h['evalue'] < existing['evalue']) or \
                       (h['evalue'] == existing['evalue'] and h['identity'] > existing['identity']):
                        best_hits_per_pdb[pid] = h

        clean_hits = list(best_hits_per_pdb.values())
        clean_hits.sort(key=lambda x: (x['evalue'], -x['identity']))

        members = subset_cluster_map.get(rep_id_full, {rep_id_full})
        for m in members:
            pdb_code = m[:4].upper()
            if pdb_code not in all_homologs:
                all_homologs[pdb_code] = {
                    "initiating_pdb": rep_pdb,
                    "hits": []
                }
            
            # Aggregate hits for this PDB entry (union)
            existing_hits_dict = {h["pdb_id"]: h for h in all_homologs[pdb_code]["hits"]}
            for h in clean_hits:
                pid = h["pdb_id"]
                if pid not in existing_hits_dict or h["evalue"] < existing_hits_dict[pid]["evalue"]:
                    existing_hits_dict[pid] = h
            
            # Update and sort
            all_homologs[pdb_code]["hits"] = sorted(existing_hits_dict.values(), key=lambda x: (x['evalue'], -x['identity']))
            
    return all_homologs

if __name__ == "__main__":
    print("Enrichment module loaded.")
