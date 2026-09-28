# main_batch.py
# A batch-aware version of main.py that accepts a configuration dictionary.

import os
import json
import sys
import math

# Add 'src' to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from ligand_detection_2D import get_phosphate_ligands_2d
from ligand_detection_3D import get_phosphate_ligands_3d
from rcsb_search import query_rcsb
from uniprot_mapping import group_pdbs_by_uniprot_id
from blast_search_standalone import run_blast_and_filter
from pdb_downloader import PDBDownloader
from tmalign_handler import TMAlignHandler
import analyze_isosteres_simple
import analyze_isosteres_advanced
import find_ligand_isosteres
import analyze_plif

def main_run(config):
    """
    Accepts a config dict to override default settings.
    Example config: {"SEARCH_COV": 0.5, "SEARCH_IDENTITY": 0.5, "DB_CLUSTER_ID": 0.95}
    """
    # 1. SETTINGS (Priority: config dict > defaults)
    LIGAND_DETECTION_MODE = config.get("LIGAND_DETECTION_MODE", "3D")
    QUERY_MODE = config.get("QUERY_MODE", "no_po4")
    RES_LIMIT = config.get("RES_LIMIT", 2.7)
    RFREE_LIMIT = config.get("RFREE_LIMIT", 0.25)
    EXPERIMENTAL_METHOD = config.get("EXPERIMENTAL_METHOD", "X-RAY DIFFRACTION")
    
    INCLUDE_MODIFIED_RESIDUES = config.get("INCLUDE_MODIFIED_RESIDUES", False)

    EVALUE_CUTOFF = config.get("EVALUE_CUTOFF", 1e-16)
    SEARCH_IDENTITY = config.get("SEARCH_IDENTITY", 0.5)
    SEARCH_COV = config.get("SEARCH_COV", 0.8)
    SEARCH_SENSITIVITY = config.get("SEARCH_SENSITIVITY", 7.5)
    DB_CLUSTER_ID = config.get("DB_CLUSTER_ID", 0.99)
    DB_CLUSTER_COV = config.get("DB_CLUSTER_COV", 0.8)

    RUN_DOWNLOAD_STEP = config.get("RUN_DOWNLOAD_STEP", True)
    RUN_TMALIGN_STEP = config.get("RUN_TMALIGN_STEP", True)
    TM_FAST_MODE = config.get("TM_FAST_MODE", True)
    EARLY_SEED_DOWNLOAD = config.get("EARLY_SEED_DOWNLOAD", True)

    # 2. GENERATE RUN ID
    q_short = "all" if QUERY_MODE == "all" else "only" if QUERY_MODE == "only_po4" else "no"
    m_short = "Xray" if EXPERIMENTAL_METHOD == "X-RAY DIFFRACTION" else "All"
    try: e_exponent = int(abs(math.log10(EVALUE_CUTOFF)))
    except: e_exponent = "var"

    run_id = (f"{LIGAND_DETECTION_MODE}_{q_short}_{RES_LIMIT}_{RFREE_LIMIT}_{m_short}_"
              f"{e_exponent}_{SEARCH_IDENTITY}_{SEARCH_COV}_{SEARCH_SENSITIVITY}_{DB_CLUSTER_ID}_{DB_CLUSTER_COV}")
    
    if TM_FAST_MODE:
        run_id += "_fast"

    print(f"\n[Batch] Starting Run: {run_id}")

    # 3. DEFINE OUTPUT PATHS
    ligand_search_file = os.path.join(PROJECT_ROOT, "results", f"ligand_search_{run_id}.json")
    final_output_file = os.path.join(PROJECT_ROOT, "results", f"homology_enrichment_{run_id}.json")
    tmalign_output_file = os.path.join(PROJECT_ROOT, "results", f"tmalign_results_{run_id}.json")

    # Initialize Downloader
    downloader = PDBDownloader(PROJECT_ROOT, max_workers=5)

    # 4. WORKFLOW LOGIC (Simplified for Batch)
    # Check if we can skip to enrichment or alignment
    if os.path.exists(final_output_file):
        print(f"  [Skip] Enrichment already exists for {run_id}.")
        with open(final_output_file, 'r') as f:
            enrichment_results = json.load(f)
    else:
        # Check if we have the ligand search
        if os.path.exists(ligand_search_file):
            print(f"  [Resume] Loading ligand search results...")
            with open(ligand_search_file, 'r') as f:
                ls_data = json.load(f)
                grouped_by_uniprot = ls_data["grouped_by_uniprot"]
                hq_pdbs = ls_data["high_quality_phosphate_pdb_codes"]
        else:
            # Full Start
            print(f"  [Step 1/2] Ligand search and RCSB query...")
            cif_path = os.path.join(PROJECT_ROOT, "data", "components.cif")
            phosphate_ligands = get_phosphate_ligands_3d(cif_path, mode=QUERY_MODE)
            hq_pdbs = query_rcsb(phosphate_ligands, mode=QUERY_MODE, min_res=RES_LIMIT, max_r_free=RFREE_LIMIT)
            grouped_by_uniprot = group_pdbs_by_uniprot_id(hq_pdbs)
            
            with open(ligand_search_file, 'w') as f:
                json.dump({"grouped_by_uniprot": grouped_by_uniprot, "high_quality_phosphate_pdb_codes": hq_pdbs}, f)

        print(f"  [Step 4] Homology Enrichment...")
        enrichment_results = run_blast_and_filter(
            grouped_by_uniprot, hq_pdbs,
            evalue=EVALUE_CUTOFF, identity=SEARCH_IDENTITY, res_limit=RES_LIMIT,
            rfree_limit=RFREE_LIMIT, method=EXPERIMENTAL_METHOD, similarity_threshold=DB_CLUSTER_ID,
            run_id=run_id, sensitivity=SEARCH_SENSITIVITY, search_cov=SEARCH_COV,
            db_cluster_cov=DB_CLUSTER_COV, downloader=downloader if EARLY_SEED_DOWNLOAD else None
        )

    # 5. DOWNLOAD & ALIGN
    if RUN_DOWNLOAD_STEP or RUN_TMALIGN_STEP:
        seeds = set(enrichment_results.get("cluster_reps", []))
        hits = set(enrichment_results.get("new_homologs_list", [])) - seeds
        
        if RUN_DOWNLOAD_STEP:
            print(f"  [Step 5] Downloading {len(seeds)} seeds...")
            downloader.download_batch(seeds, category="references")

        if RUN_TMALIGN_STEP:
            if os.path.exists(tmalign_output_file):
                print(f"  [Skip] TM-align results already exist.")
            else:
                print(f"  [Step 6] TM-align for {len(hits)} hits (Fast Mode: {TM_FAST_MODE})...")
                tm_handler = TMAlignHandler(PROJECT_ROOT, fast_mode=TM_FAST_MODE)
                tm_handler.set_checkpoint(tmalign_output_file)
                
                # Setup callback for concurrent download/align
                hit_to_seed = {}
                for seed, data in enrichment_results["cluster_hits"].items():
                    for hit in data["hits"]:
                        hit_to_seed[hit["pdb_id"].upper()] = seed.upper()

                with tm_handler:
                    if RUN_DOWNLOAD_STEP:
                        downloader.download_batch(hits, category="hits", on_downloaded=lambda pid: tm_handler.submit_alignment(pid, hit_to_seed.get(pid.upper())))
                    else:
                        for hit_pdb in hits:
                            seed_pdb = hit_to_seed.get(hit_pdb.upper())
                            if seed_pdb: tm_handler.submit_alignment(hit_pdb, seed_pdb)
                    
                    tm_handler.collect_results(wait_for_all=True)
                print(f"  [Done] TM-align results saved.")

    # 6. ANALYSIS
    if RUN_TMALIGN_STEP:
        print("\n" + "="*80)
        print("STEP 7: ANALYZING RESULTS FOR ISOSTERES")
        print("-" * 80)
        motif_dir = os.path.join(PROJECT_ROOT, "results", "motif_analysis")
        
        analyze_isosteres_simple.analyze_isosteres_simple(motif_dir, include_modified=INCLUDE_MODIFIED_RESIDUES)
        analyze_isosteres_advanced.analyze_isosteres(motif_dir, include_modified=INCLUDE_MODIFIED_RESIDUES)
        find_ligand_isosteres.find_ligand_isosteres(motif_dir, include_modified=INCLUDE_MODIFIED_RESIDUES)
        analyze_plif.run_plif_analysis(motif_dir)
        
        print("="*80 + "\n")

if __name__ == "__main__":
    # Test with a single default run if called directly
    main_run({})
