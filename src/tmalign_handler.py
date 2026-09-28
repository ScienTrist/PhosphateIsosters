import os
import json
import time
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from tmalign_wrapper import run_tmalign

class TMAlignHandler:
    def __init__(self, project_root, tmalign_executable=None, max_workers=None, fast_mode=True):
        self.project_root = project_root
        self.fast_mode = fast_mode
        
        # Default workers: Be more conservative on Windows due to WSL overhead
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            if platform.system() == "Windows":
                self.max_workers = min(cpu_count, 8) # Avoid overwhelming WSL
            else:
                self.max_workers = cpu_count
        else:
            self.max_workers = max_workers
        
        # Default executable location
        if tmalign_executable is None:
            exec_dir = os.path.join(project_root, "scripts", "Shell scripts")
            if platform.system() == "Linux":
                if os.path.exists(os.path.join(exec_dir, "TMalign_linux")):
                    self.tmalign_executable = os.path.join(exec_dir, "TMalign_linux")
                else:
                    self.tmalign_executable = os.path.join(exec_dir, "TMalign")
            else:
                self.tmalign_executable = os.path.join(exec_dir, "TMalign")
        else:
            self.tmalign_executable = tmalign_executable
            
        self.results_dir = os.path.join(project_root, "results")
        self.ref_dir = os.path.join(project_root, "data", "structures", "references")
        self.hits_dir = os.path.join(project_root, "data", "structures", "hits")

        # For incremental/persistent execution
        self._executor = None
        self._futures = {} # Maps future to (hit_pdb, seed_pdb)
        self._results = {}
        self._processed_count = 0
        self._start_time = None
        self._checkpoint_path = None

    def __enter__(self):
        print(f"  [TM-align] Initializing worker pool with {self.max_workers} processes...")
        self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _get_executor(self):
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def set_checkpoint(self, path):
        """Optional: Load existing results and set path for auto-saving."""
        self._checkpoint_path = path
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    self._results = json.load(f)
                # Count already processed hits
                count = sum(len(hits) for hits in self._results.values())
                print(f"  [TM-align] Resuming from checkpoint: {count} alignments already completed.")
            except Exception as e:
                print(f"  [TM-align] Warning: Could not load checkpoint: {e}")

    def submit_alignment(self, hit_pdb, seed_pdb):
        """Queues a single alignment task, skipping if already in results."""
        # Check if already done
        if seed_pdb in self._results:
            if any(r.get("hit") == hit_pdb for r in self._results[seed_pdb]):
                return None

        if self._start_time is None:
            self._start_time = time.time()
            
        executor = self._get_executor()
        future = executor.submit(
            run_tmalign_task, 
            hit_pdb, 
            seed_pdb, 
            self.ref_dir, 
            self.hits_dir, 
            self.tmalign_executable,
            self.fast_mode
        )
        self._futures[future] = (hit_pdb, seed_pdb)
        
        # Periodically collect results if the queue is getting too large
        # This prevents the "thousands of futures" RAM issue
        if len(self._futures) > 500:
            self.collect_results(wait_for_all=False)
            
        return future

    def collect_results(self, wait_for_all=True):
        """
        Gathers completed results. 
        If wait_for_all is False, only processes what's already finished.
        """
        if not self._futures:
            return self._results

        # CRITICAL: We take a snapshot of the keys to avoid "dictionary changed size"
        # However, as_completed needs the actual future objects.
        all_current_futures = list(self._futures.keys())
        futures_to_check = as_completed(all_current_futures) if wait_for_all else []
        
        # If not waiting, we manually check which futures are done
        if not wait_for_all:
            completed = [f for f in self._futures.keys() if f.done()]
            if not completed:
                return self._results
            futures_to_check = completed

        if wait_for_all:
            total_tasks = self._processed_count + len(self._futures)
            print(f"  [TM-align] Finalizing {len(self._futures)} remaining alignment tasks...")

        for future in futures_to_check:
            try:
                hit_pdb, seed_pdb, alignment, status = future.result()
                self._processed_count += 1
                
                if status == "success":
                    if seed_pdb not in self._results:
                        self._results[seed_pdb] = []
                    # Avoid duplicates
                    if not any(r.get("hit") == hit_pdb for r in self._results[seed_pdb]):
                        self._results[seed_pdb].append(alignment)
                
                # Progress update every 50 alignments
                if wait_for_all and (self._processed_count % 50 == 0 or self._processed_count == total_tasks):
                    elapsed = time.time() - self._start_time
                    avg = elapsed / self._processed_count
                    remaining = (total_tasks - self._processed_count) * avg
                    print(f"\r    [TM-align] Progress: {self._processed_count}/{total_tasks} aligned... Est. remaining: {remaining/60:.1f}m", end="", flush=True)

                # Periodic Checkpoint Saving (every 100)
                if self._processed_count % 100 == 0 and self._checkpoint_path:
                    self._save_checkpoint()

            except Exception as e:
                print(f"\n  [Error] Alignment task failed: {e}")
            finally:
                # CRITICAL: Remove future from memory as soon as it's processed
                if future in self._futures:
                    del self._futures[future]

        if wait_for_all:
            print() # New line
            self._save_checkpoint() # Final save
            
        return self._results

    def _save_checkpoint(self):
        """Saves current results to the checkpoint file."""
        if not self._checkpoint_path:
            return
            
        try:
            temp_path = self._checkpoint_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(self._results, f, indent=2)
            
            if os.path.exists(self._checkpoint_path):
                os.remove(self._checkpoint_path)
            os.rename(temp_path, self._checkpoint_path)
        except Exception as e:
            print(f"\n  [Warning] Failed to save TM-align checkpoint: {e}")

    def run_batch_alignment(self, cluster_results, run_id):
        """Runs parallel alignments for all hits in the cluster results."""
        tasks = []
        for seed_pdb, data in cluster_results.items():
            for hit in data["hits"]:
                hit_pdb = hit["pdb_id"]
                tasks.append((hit_pdb, seed_pdb))
        
        if not tasks:
            print("  [TM-align] No hits found to align.")
            return {}

        print(f"  [TM-align] Starting batch alignment of {len(tasks)} structures using {self.max_workers} processes...")
        
        # Reset state for this batch
        self._results = {}
        self._processed_count = 0
        self._start_time = time.time()
        self.set_checkpoint(os.path.join(self.results_dir, f"tmalign_results_{run_id}.json"))

        with self: 
            for hit_pdb, seed_pdb in tasks:
                self.submit_alignment(hit_pdb, seed_pdb)
            self.collect_results(wait_for_all=True)
            
        print(f"  [TM-align] Completed! Results saved to {self._checkpoint_path}")
        return self._results

def run_tmalign_task(hit_pdb, seed_pdb, ref_dir, hits_dir, executable, fast_mode=True):
    """Stand-alone function for process pooling (must be picklable)."""
    from tmalign_wrapper import run_tmalign
    from utils import find_structure_in_dir as find_file

    hit_path = find_file(hit_pdb, hits_dir)
    seed_path = find_file(seed_pdb, ref_dir)
    
    if not hit_path or not seed_path:
        return hit_pdb, seed_pdb, None, "missing_files"
        
    try:
        res = run_tmalign(hit_path, seed_path, tmalign_executable=executable, fast_mode=fast_mode)
        if res:
            res.pop("raw_output", None)
            res["hit"] = hit_pdb
            return hit_pdb, seed_pdb, res, "success"
    except Exception:
        pass
        
    return hit_pdb, seed_pdb, None, "failed"

def run_tmalign_workflow(cluster_results, run_id, fast_mode=True):
    """Helper for main.py."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    handler = TMAlignHandler(project_root, fast_mode=fast_mode)
    return handler.run_batch_alignment(cluster_results, run_id)
