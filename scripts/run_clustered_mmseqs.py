import json
import os
import subprocess

# Paths
ROOT = "/mnt/d/AI/Master_thesis/Phosphate-binding-site"
SEQRES = os.path.join(ROOT, "data/pdb_seqres.txt")
JSON_CACHE = os.path.join(ROOT, "cache/blast_cache_1e-16_0.5_0.95.json")
SAMPLE_KEYS_FILE = os.path.join(ROOT, "data/sample_500_keys.txt")

MMSEQS_BIN = "/home/trist/.local/lib/python3.12/site-packages/pymmseqs/bin/mmseqs"
RESULTS_DIR = os.path.join(ROOT, "results/clustered_mmseqs")
TMP_DIR = os.path.join(RESULTS_DIR, "tmp")

ALL_PROTEINS_FASTA = os.path.join(ROOT, "data/pdb_proteins_all.fasta")
CLUSTER_PREFIX = os.path.join(RESULTS_DIR, "pdb_clustered")
# mmseqs easy-cluster with these settings produced:
REP_FASTA = os.path.join(RESULTS_DIR, "pdb_clustered_rep_seq.fasta")
QUERY_FASTA = os.path.join(ROOT, "data/sample_500.fasta")
RESULT_TSV = os.path.join(RESULTS_DIR, "mmseqs_clustered_results.tsv")

def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def setup():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

def prepare_all_proteins():
    if os.path.exists(ALL_PROTEINS_FASTA):
        print(f"{ALL_PROTEINS_FASTA} already exists, skipping...")
        return
    print("Extracting all proteins from pdb_seqres.txt...")
    with open(SEQRES, 'r', encoding='utf-8', errors='ignore') as f, open(ALL_PROTEINS_FASTA, 'w', encoding='utf-8') as tf:
        current_header = None
        for line in f:
            if line.startswith(">"):
                current_header = line.strip()
            else:
                seq = line.strip()
                if current_header and "mol:protein" in current_header:
                    # Keep chain ID in header for better mapping
                    tf.write(f"{current_header}\n{seq}\n")

def cluster_database():
    print("Step 1: Clustering the full database at 0.95 identity...")
    # mmseqs easy-cluster input output tmp --min-seq-id 0.95 -c 0.8
    # We use -c 0.8 as a standard coverage for clustering, but 0.95 identity as requested
    cmd = [
        MMSEQS_BIN, "easy-cluster",
        ALL_PROTEINS_FASTA,
        CLUSTER_PREFIX,
        TMP_DIR,
        "--min-seq-id", "0.95",
        "-c", "0.8",
        "--threads", "16"
    ]
    # Check if the output _rep file exists
    if not os.path.exists(REP_FASTA):
        run_cmd(cmd)
    else:
        print(f"{REP_FASTA} already exists, skipping...")

def perform_search():
    print("Step 2: Performing search against representatives (Identity 0.5, E-value 1e-16)...")
    # Using easy-search against the representatives
    # Note: --min-seq-id can be 0.0-1.0
    cmd = [
        MMSEQS_BIN, "easy-search",
        QUERY_FASTA,
        REP_FASTA, # Searching ONLY against cluster representatives
        RESULT_TSV,
        TMP_DIR,
        "-e", "1e-16",
        "--min-seq-id", "0.5", 
        "-c", "0.5", 
        "-s", "7.5",
        "--threads", "16"
    ]
    run_cmd(cmd)

def compare_and_report():
    print("Comparing results with cluster expansion...")
    with open(JSON_CACHE, 'r') as f:
        blast_data = json.load(f)
    
    with open(SAMPLE_KEYS_FILE, 'r', encoding='utf-8') as f:
        sample_keys = [line.strip() for line in f]

    # Load cluster assignments
    # pdb_clustered_cluster.tsv has: representative\tmember
    print("Loading cluster assignments...")
    cluster_to_members = {}
    cluster_tsv = os.path.join(RESULTS_DIR, "pdb_clustered_cluster.tsv")
    with open(cluster_tsv, 'r') as f:
        for line in f:
            rep, member = line.strip().split("\t")
            rep_pdb = rep[:4].upper()
            member_pdb = member[:4].upper()
            if rep_pdb not in cluster_to_members:
                cluster_to_members[rep_pdb] = set()
            cluster_to_members[rep_pdb].add(member_pdb)

    mmseqs_hits_expanded = {}
    if os.path.exists(RESULT_TSV):
        with open(RESULT_TSV, 'r') as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    query = parts[0]
                    target_rep_pdb = parts[1][:4].upper()
                    
                    if query not in mmseqs_hits_expanded:
                        mmseqs_hits_expanded[query] = set()
                    
                    # Expand hit to all members of the cluster
                    members = cluster_to_members.get(target_rep_pdb, {target_rep_pdb})
                    mmseqs_hits_expanded[query].update(members)

    total_blast = 0
    total_mmseqs_expanded = 0
    total_overlap = 0
    
    print(f"\nSummary for {len(sample_keys)} queries (Expanded Clusters):")
    for key in sample_keys:
        blast_hits = set(h['pdb_id'].upper() for h in blast_data.get(key, {}).get('hits', []))
        m_hits = mmseqs_hits_expanded.get(key, set())
        
        total_blast += len(blast_hits)
        total_mmseqs_expanded += len(m_hits)
        total_overlap += len(blast_hits.intersection(m_hits))

    print(f"Total BLAST hits:           {total_blast}")
    print(f"Total MMseqs expanded hits: {total_mmseqs_expanded}")
    print(f"Total Overlap:              {total_overlap}")
    if total_blast > 0:
        print(f"Recall:                     {total_overlap/total_blast:.2%}")
    if total_mmseqs_expanded > 0:
        print(f"Precision-like:             {total_overlap/total_mmseqs_expanded:.2%}")

if __name__ == "__main__":
    setup()
    prepare_all_proteins()
    cluster_database()
    perform_search()
    compare_and_report()
