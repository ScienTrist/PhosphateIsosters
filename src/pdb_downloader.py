import os
import json
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class PDBDownloader:
    def __init__(self, project_root, max_workers=5):
        self.project_root = project_root
        self.max_workers = max_workers
        self.ref_dir = os.path.join(project_root, "data", "structures", "references")
        self.hits_dir = os.path.join(project_root, "data", "structures", "hits")
        self.manifest_path = os.path.join(project_root, "cache", "download_manifest.json")
        self.current_thread = None
        
        # Thread safety for the manifest dictionary
        self.manifest_lock = threading.Lock()
        
        # Persistent Session for connection pooling (HUGE speed gain)
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers, 
            pool_maxsize=max_workers,
            max_retries=3
        )
        self.session.mount("https://", adapter)
        
        # Ensure directories exist
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.hits_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        
        self.manifest = self._load_manifest()
        
        # If manifest is empty (e.g. first run or deleted), scan local folders
        # to avoid re-downloading thousands of files
        if not self.manifest:
            print("  [Downloader] Manifest empty. Scanning local directories for existing files...")
            self._scan_local_files()
            if self.manifest:
                print(f"  [Downloader] Found {len(self.manifest)} files locally. Manifest updated.")
                self._save_manifest()

    def _scan_local_files(self):
        """Scans the references and hits directories to populate the manifest with existing files."""
        for directory in [self.ref_dir, self.hits_dir]:
            if not os.path.exists(directory):
                continue
                
            for filename in os.listdir(directory):
                if filename.endswith((".pdb", ".cif")):
                    pdb_id = filename.split(".")[0].upper()
                    if pdb_id not in self.manifest:
                        self.manifest[pdb_id] = os.path.join(directory, filename)

    def _load_manifest(self):
        """Loads the manifest from disk. Done only at initialization."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r') as f:
                    rel_manifest = json.load(f)
                    # Convert relative paths from manifest back to absolute paths
                    # This ensures compatibility between Windows and WSL
                    with self.manifest_lock:
                        return {k: os.path.join(self.project_root, v) if not os.path.isabs(v) else v 
                                for k, v in rel_manifest.items()}
            except:
                return {}
        return {}

    def _save_manifest(self):
        """Saves the manifest atomically to prevent corruption using relative paths."""
        temp_path = self.manifest_path + ".tmp"
        try:
            # Thread-safe copy to avoid "dictionary changed size during iteration"
            with self.manifest_lock:
                manifest_copy = self.manifest.copy()

            # Convert paths to relative to project_root for cross-platform compatibility
            rel_manifest = {}
            for k, v in manifest_copy.items():
                try:
                    rel_path = os.path.relpath(v, self.project_root)
                    rel_manifest[k] = rel_path
                except ValueError:
                    # Fallback if path is on a different drive (Windows)
                    rel_manifest[k] = v

            with open(temp_path, 'w') as f:
                json.dump(rel_manifest, f, indent=2)

            # Atomic rename (on Windows we must remove old file first)
            if os.path.exists(self.manifest_path):
                os.remove(self.manifest_path)
            os.rename(temp_path, self.manifest_path)
        except Exception as e:
            # We don't want a save error to crash the whole download loop
            print(f"\n    [Error] Failed to save manifest: {e}")
    def _download_single(self, pdb_id, target_dir):
        """Downloads a single PDB ID as .pdb.gz, falling back to .cif.gz if unavailable."""
        pdb_id = pdb_id.upper()

        # Check manifest first
        with self.manifest_lock:
            if pdb_id in self.manifest and os.path.exists(self.manifest[pdb_id]):
                return pdb_id, "cached"

        import gzip
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
        pdb_path = os.path.join(target_dir, f"{pdb_id}.pdb")
        
        try:
            response = self.session.get(pdb_url, timeout=15)
            if response.status_code == 200:
                # Decompress on the fly
                content = gzip.decompress(response.content)
                with open(pdb_path, 'wb') as f:
                    f.write(content)
                # Thread-safe write
                with self.manifest_lock:
                    self.manifest[pdb_id] = pdb_path
                return pdb_id, "downloaded (.pdb.gz)"
            
            # 3. Fallback to CIF if PDB is not available (usually 404 for large structures)
            elif response.status_code == 404:
                cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
                cif_path = os.path.join(target_dir, f"{pdb_id}.cif")
                
                cif_response = self.session.get(cif_url, timeout=15)
                if cif_response.status_code == 200:
                    # Decompress on the fly
                    content = gzip.decompress(cif_response.content)
                    with open(cif_path, 'wb') as f:
                        f.write(content)
                    # Thread-safe write
                    with self.manifest_lock:
                        self.manifest[pdb_id] = cif_path
                    return pdb_id, "downloaded (.cif.gz fallback)"
                else:
                    return pdb_id, f"failed cif (status {cif_response.status_code})"
            else:
                return pdb_id, f"failed pdb (status {response.status_code})"
                
        except Exception as e:
            return pdb_id, f"error ({str(e)})"

    def download_batch(self, pdb_ids, category="hits", label=None, on_downloaded=None):
        """Downloads a batch of PDB IDs into the specified category directory."""
        
        # Wait for any previous background download to finish to avoid manifest conflicts
        # But ONLY if we are NOT in that background thread ourselves!
        if self.current_thread and self.current_thread.is_alive() and self.current_thread != threading.current_thread():
            print(f"\n  [Downloader] Waiting for early seed download to complete...")
            self.current_thread.join()
            print(f"  [Downloader] Early download finished. Proceeding with {category} batch.")

        target_dir = self.ref_dir if category == "references" else self.hits_dir
        display_label = label if label else f"[{category.capitalize()}]"
        
        # 1. PRE-FILTER: Instantly skip anything already in the manifest
        all_ids = set(pid.upper() for pid in pdb_ids)
        to_download = []
        cached_ids = []
        
        with self.manifest_lock:
            for pid in all_ids:
                if pid in self.manifest and os.path.exists(self.manifest[pid]):
                    cached_ids.append(pid)
                else:
                    to_download.append(pid)
        
        cached_count = len(cached_ids)
        total_to_do = len(to_download)
        print(f"  [Downloader] {category.capitalize()}: {cached_count} already cached, {total_to_do} to process.")
        
        # Trigger callback for cached files immediately
        if on_downloaded:
            for pid in cached_ids:
                on_downloaded(pid)

        results = {"downloaded": 0, "cached": cached_count, "failed": 0}
        if total_to_do == 0:
            return results

        processed = 0
        checkpoint_interval = 50 # Save manifest every 50 files
        start_time = time.time()

        # 2. DOWNLOAD POOL
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pdb = {executor.submit(self._download_single, pid, target_dir): pid for pid in to_download}

            for future in as_completed(future_to_pdb):
                pdb_id, status = future.result()
                processed += 1

                if "downloaded" in status:
                    results["downloaded"] += 1
                else:
                    results["failed"] += 1
                    print(f"\n    [Warning] Failed to download {pdb_id}: {status}")

                # Live progress update with ETA
                elapsed = time.time() - start_time
                avg = elapsed / processed
                remaining = (total_to_do - processed) * avg

                progress_str = f"\r    {display_label} Progress: {processed}/{total_to_do} processed... Est. remaining: {remaining/60:.1f}m"
                print(progress_str.ljust(90), end="", flush=True)

                if on_downloaded and "downloaded" in status:
                    on_downloaded(pdb_id)

                # Checkpoint saving: Save progress mid-run
                if processed % checkpoint_interval == 0:
                    self._save_manifest()

        print() # New line after finishing
        self._save_manifest() # Final save
        return results

    def start_background_download(self, pdb_ids, category="hits"):
        """Starts a download batch in a separate thread."""
        self.current_thread = threading.Thread(
            target=self.download_batch, 
            args=(pdb_ids,), 
            kwargs={"category": category, "label": "[Background Seeds]"}
        )
        self.current_thread.daemon = True # Ensure it doesn't block program exit
        self.current_thread.start()
        return self.current_thread

def run_batch_download(enrichment_results, downloader=None):
    """
    Helper function to be called from main.py.
    Takes the output of run_homology_enrichment.
    """
    if downloader is None:
        # Identify project root relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        downloader = PDBDownloader(project_root)
    
    # 1. Collect Seed PDBs (The unique cluster representatives)
    seeds = set(enrichment_results.get("cluster_reps", []))
    
    # 2. Collect Hit PDBs (The newly found homologs)
    hits = set(enrichment_results.get("new_homologs_list", []))
    
    # Ensure no duplicates between categories (Seeds take priority for organization)
    hits = hits - seeds
    
    print("-" * 30)
    print("Step 5: Batch Downloading Structure Files...")
    print(f"  Target: {len(seeds)} Seeds (Representatives) and {len(hits)} New Homologs.")
    
    seed_res = downloader.download_batch(seeds, category="references")
    hit_res = downloader.download_batch(hits, category="hits")
    
    print(f"  Seeds (Ref): {seed_res['downloaded']} new, {seed_res['cached']} cached")
    print(f"  Hits:        {hit_res['downloaded']} new, {hit_res['cached']} cached")
    if seed_res['failed'] > 0 or hit_res['failed'] > 0:
        print(f"  [Notice] {seed_res['failed'] + hit_res['failed']} files failed to download.")
    print("-" * 30)
