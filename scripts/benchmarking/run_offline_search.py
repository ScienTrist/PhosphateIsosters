import os
import subprocess
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from utils import to_wsl_path

# Define Paths
ROOT = "Master_thesis/Phosphate-binding-site"
QUERY_FASTA = os.path.join(ROOT, "data/sample_500.fasta")
# USE CLUSTERED REPRESENTATIVES FOR SEARCH
TARGET_FASTA = os.path.join(ROOT, "results/clustered_mmseqs/pdb_clustered_rep_seq.fasta")
RESULT_TSV = os.path.join(ROOT, "results/benchmarking/temp_results_500.tsv")
TMP_DIR = os.path.join(ROOT, "results/benchmarking/tmp_mmseqs")

# Convert Windows paths to WSL paths (e.g. D:\... -> /mnt/d/...)
def to_wsl(path):
    return to_wsl_path(os.path.abspath(path))

WSL_QUERY = to_wsl(QUERY_FASTA)
WSL_TARGET = to_wsl(TARGET_FASTA)
WSL_RESULT = to_wsl(RESULT_TSV)
WSL_TMP = to_wsl(TMP_DIR)

MMSEQS_BINARY = "/home/trist/.local/lib/python3.12/site-packages/pymmseqs/bin/mmseqs"

def run_search():
    print(f"Ensuring TMP_DIR exists: {TMP_DIR}")
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)
    
    print("Running MMseqs2 easy-search against CLUSTERED REPS via WSL...")
    cmd = [
        "wsl",
        MMSEQS_BINARY, "easy-search",
        WSL_QUERY,
        WSL_TARGET,
        WSL_RESULT,
        WSL_TMP,
        "-e", "1e-16",
        "--min-seq-id", "0.5",
        "-c", "0.5",
        "-s", "7.5",
        "--threads", "16"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("MMseqs2 search completed successfully.")
        print(f"Results saved to: {RESULT_TSV}")
    else:
        print("MMseqs2 search failed.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    
    # Cleanup
    if os.path.exists(TMP_DIR):
        print(f"Cleaning up temporary directory: {TMP_DIR}")
        shutil.rmtree(TMP_DIR)

if __name__ == "__main__":
    run_search()
