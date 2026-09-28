import os
import subprocess
import platform
import json
import glob
import time
import shutil

class MMseqsHandler:
    def __init__(self, project_root, mmseqs_bin="/home/trist/.local/lib/python3.12/site-packages/pymmseqs/bin/mmseqs", threads=None):
        self.project_root = project_root
        self.mmseqs_bin = mmseqs_bin
        
        # Automatically detect optimal thread count if not specified
        if threads is None:
            auto_threads = os.cpu_count() or 1
            # For MMseqs2, we cap it at 16 or 32 for efficiency, but using all cores is generally fine
            self.threads = auto_threads
        else:
            self.threads = threads

        self.is_windows = (platform.system() == "Windows")
        
        self.tmp_dir = os.path.join(self.project_root, "results", "tmp_mmseqs")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def to_wsl(self, path):
        if not self.is_windows or not path:
            return path
        if ":" in path:
            drive, rest = path.split(":", 1)
            return f"/mnt/{drive.lower()}{rest.replace('\\', '/')}"
        return path.replace("\\", "/")

    def run_cmd(self, cmd, desc="MMseqs2 Task"):
        if self.is_windows:
            full_cmd = ["wsl"] + cmd
        else:
            full_cmd = cmd
        
        print(f"  [Task] {desc}...")
        
        process = subprocess.Popen(
            full_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )

        full_output = []
        current_stage = ""
        start_time = time.time()
        
        try:
            for line in process.stdout:
                full_output.append(line)
                clean_line = line.strip()
                
                # Detect Stage Transitions (MMseqs2 modules)
                # Examples: "prefilter ...", "align ...", "result2flat ..."
                if any(stage in clean_line.lower() for stage in ["prefilter", "align", "createdb", "cluster", "createtsv", "result2repseq"]):
                    if "..." in clean_line and not clean_line.startswith("["):
                        new_stage = clean_line.split("...")[0].strip().capitalize()
                        if new_stage != current_stage:
                            current_stage = new_stage
                            print(f"\n    [MMseqs2] {current_stage} stage started...")

                # Parse Progress Bar
                # Example: [========>] 10.5K 0s
                if "[" in clean_line and "=" in clean_line:
                    parts = clean_line.split("]")
                    if len(parts) > 1:
                        # Extract the count and time info
                        # Example: " 10.5K 1s"
                        progress_info = parts[1].strip()
                        elapsed = int(time.time() - start_time)
                        
                        # Live progress update - padded with spaces to clear the line
                        # We use \r to stay on the same line
                        progress_str = f"\r    {current_stage:12} | Progress: {progress_info} | Total Elapsed: {elapsed}s"
                        print(progress_str.ljust(80), end="", flush=True)
                
                # Handle other informative lines
                elif "Number of clusters" in clean_line:
                    print(f"\n    {clean_line}", flush=True)
                elif "search" in clean_line.lower() and "complete" in clean_line.lower():
                    print(f"\n    {clean_line}", flush=True)

            process.wait()
        except KeyboardInterrupt:
            print("\n\n!!! INTERRUPTED BY USER !!!")
            process.terminate()
            self._dump_diagnostics(full_output)
            raise

        if process.returncode != 0:
            print(f"\n\n!!! {desc} FAILED (Exit Code {process.returncode}) !!!")
            self._dump_diagnostics(full_output)
            return False
        return True

    def _dump_diagnostics(self, captured_stdout):
        print("="*80)
        print("FULL OUTPUT FROM FAILED STEP:")
        print("-" * 80)
        print("".join(captured_stdout))
        
        log_files = glob.glob(os.path.join(self.tmp_dir, "**", "*.log"), recursive=True)
        if log_files:
            log_files.sort(key=os.path.getmtime, reverse=True)
            latest_log = log_files[0]
            print(f"\nINTERNAL LOG FILE: {latest_log}")
            try:
                with open(latest_log, 'r') as f:
                    print("".join(f.readlines()[-50:]))
            except: pass
        print("="*80)

    def cluster_database(self, fasta_path, output_prefix, min_seq_id=0.95, coverage=0.8, cluster_mode=0, cov_mode=0):
        """Replaces easy-cluster with individual steps to find the exact failure point."""
        
        # CLEANUP before starting
        if os.path.exists(self.tmp_dir):
            for item in os.listdir(self.tmp_dir):
                item_path = os.path.join(self.tmp_dir, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"    [Warning] Could not delete {item_path}: {e}")

        wsl_fasta = self.to_wsl(os.path.abspath(fasta_path))
        wsl_out = self.to_wsl(os.path.abspath(output_prefix))
        wsl_tmp = self.to_wsl(os.path.abspath(self.tmp_dir))
        
        # Define internal DB paths
        db_base = os.path.join(self.tmp_dir, "db")
        wsl_db = self.to_wsl(db_base)
        
        # Step 1: Create DB
        if not self.run_cmd([self.mmseqs_bin, "createdb", wsl_fasta, wsl_db], "Create Database"):
            return False
            
        # Step 2: Cluster
        clu_db = os.path.join(self.tmp_dir, "clu")
        wsl_clu = self.to_wsl(clu_db)
        # Using --cluster-mode 0 (Set Cover) by default as it is RCSB standard
        if not self.run_cmd([self.mmseqs_bin, "cluster", wsl_db, wsl_clu, wsl_tmp, 
                            "--min-seq-id", str(min_seq_id), "-c", str(coverage), 
                            "--cluster-mode", str(cluster_mode), "--cov-mode", str(cov_mode),
                            "--threads", str(self.threads)], "Clustering"):
            return False
            
        # Step 3: Create TSV (This is the mapping file)
        wsl_tsv = self.to_wsl(os.path.abspath(output_prefix + "_cluster.tsv"))
        if not self.run_cmd([self.mmseqs_bin, "createtsv", wsl_db, wsl_db, wsl_clu, wsl_tsv], "Generate TSV Map"):
            return False
            
        # Step 4: result2repseq (Identify representatives)
        rep_db = os.path.join(self.tmp_dir, "rep")
        wsl_rep_db = self.to_wsl(rep_db)
        if not self.run_cmd([self.mmseqs_bin, "result2repseq", wsl_db, wsl_clu, wsl_rep_db], "Identify Representatives"):
            return False
            
        # Step 5: result2flat (Extract representatives to FASTA)
        wsl_rep_fasta = self.to_wsl(os.path.abspath(output_prefix + "_rep_seq.fasta"))
        if not self.run_cmd([self.mmseqs_bin, "result2flat", wsl_db, wsl_db, wsl_rep_db, wsl_rep_fasta], "Extract FASTA"):
            return False
            
        return True

    def search(self, query_fasta, target_fasta, result_tsv, min_seq_id=0.5, evalue=1e-16, coverage=0.5, sensitivity=7.5):
        wsl_query = self.to_wsl(os.path.abspath(query_fasta))
        wsl_target = self.to_wsl(os.path.abspath(target_fasta))
        wsl_res = self.to_wsl(os.path.abspath(result_tsv))
        wsl_tmp = self.to_wsl(os.path.abspath(self.tmp_dir))
        
        # We increase --max-seqs to 20,000 to ensure we don't miss hits for very common proteins (like polymerases)
        # We also set --max-accept to a high value to ensure all hits passing the e-value/identity are returned.
        cmd = [
            self.mmseqs_bin, "easy-search",
            wsl_query, wsl_target, wsl_res, wsl_tmp,
            "--min-seq-id", str(min_seq_id), "-e", str(evalue), "-c", str(coverage), "--threads", str(self.threads),
            "-s", str(sensitivity), "--max-seqs", "20000", "--max-accept", "20000"
        ]
        return self.run_cmd(cmd, "Homology Search")

    def load_cluster_map(self, cluster_tsv):
        cluster_map = {}
        if not os.path.exists(cluster_tsv): return cluster_map
        with open(cluster_tsv, 'r') as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) == 2:
                    if p[0] not in cluster_map: cluster_map[p[0]] = set()
                    cluster_map[p[0]].add(p[1])
        return cluster_map

    def parse_search_results(self, result_tsv):
        hits = {}
        if not os.path.exists(result_tsv): return hits
        with open(result_tsv, 'r') as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 11:
                    q, t = p[0], p[1]
                    raw_ident = float(p[2])
                    # If identity is > 1.0, it's likely a percentage (0-100), so divide by 100.
                    # Otherwise, assume it's already a fraction (0.0-1.0).
                    ident = raw_ident / 100.0 if raw_ident > 1.0 else raw_ident
                    ev = float(p[10])
                    
                    if q not in hits: hits[q] = []
                    hits[q].append({"pdb_id": t[:4].upper(), "full_id": t, "identity": ident, "evalue": ev})
        return hits
    
    def clean_up_run(self, tag=None):
        """Deletes the entire temporary directory and its contents."""
        if os.path.exists(self.tmp_dir):
            print(f"  [Cleanup] Removing temporary MMseqs2 directory: {self.tmp_dir}")
            try:
                if self.is_windows:
                    # On Windows, MMseqs2 (via WSL) creates Linux symlinks that shutil.rmtree 
                    # cannot handle (WinError 1920). We use WSL to delete them.
                    wsl_tmp = self.to_wsl(os.path.abspath(self.tmp_dir))
                    subprocess.run(["wsl", "rm", "-rf", wsl_tmp], check=True)
                    # Recreate the directory
                    os.makedirs(self.tmp_dir)
                else:
                    shutil.rmtree(self.tmp_dir)
                    os.makedirs(self.tmp_dir)
            except Exception as e:
                print(f"    [Error] Cleanup failed: {e}")
