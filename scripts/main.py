# main.py
# This script serves as the central entry point for the PDB query workflow.

import os
import json
import sys

# Add 'src' to path so we can import our modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Import the refactored functions from the other modules
from ligand_detection_2D import get_phosphate_ligands_2d
from ligand_detection_3D import get_phosphate_ligands_3d
from rcsb_search import query_rcsb, filter_by_quality
from uniprot_mapping import group_pdbs_by_uniprot_id
from blast_search_standalone import run_blast_and_filter
from pdb_downloader import run_batch_download
from tmalign_handler import run_tmalign_workflow
import analyze_isosteres_simple
import analyze_plif
import sort_hits as sort_hits_module
import export_to_datawarrior
import batch_motif_extraction

def main():
    """
    Orchestrates the entire workflow:
    1. Identifies phosphate-containing ligands (2D, 3D, or Combined).
    2. Queries the RCSB database for PDB structures containing these ligands.
    3. Groups the resulting PDB IDs by their corresponding UniProt ID.
    4. Saves intermediate results to JSON.
    5. Runs BLAST enrichment for homologs with user-defined settings.
    6. Downloads PDB/CIF structure files for analysis.
    7. Performs structural alignment via TM-align.
    """
    # CONFIGURATION: 
    # Choose ligand detection mode ("2D", "3D", or "2D+3D")
    LIGAND_DETECTION_MODE = "3D"

    # DOWNLOAD SETTINGS
    RUN_DOWNLOAD_STEP = True   # Set to False to skip structure downloads
    EARLY_SEED_DOWNLOAD = True  # If True, starts downloading seed structures while BLAST search is running
    
    # TM-ALIGN SETTINGS
    RUN_TMALIGN_STEP = True   # Set to False to skip structural alignment
    TM_FAST_MODE = False      # Set to True for 10x speedup with minimal accuracy loss

    # Choose which RCSB query mode to run ("all", "only_po4", or "no_po4")
    # "only_po4": isolates simple phosphates (PO4, PI)
    # "no_po4":   excludes structures containing ONLY simple phosphates
    QUERY_MODE = "no_po4"

    # QUALITY CRITERIA
    RES_LIMIT = 2.7
    RFREE_LIMIT = 0.25
    EXPERIMENTAL_METHOD = "X-RAY DIFFRACTION"  # Options: "X-RAY DIFFRACTION" or "ALL"

    # ANALYSIS SETTINGS
    INCLUDE_MODIFIED_RESIDUES = False  # Set to True to include modified residues as mimics in hits

    # MMSEQS2 / HOMOLOGY SETTINGS
    EVALUE_CUTOFF = 1e-16
    SEARCH_IDENTITY = 0.5    # Identity for finding homologs
    SEARCH_COV = 0.8         # Coverage for finding homologs (0.8 for strict global alignment)
    SEARCH_SENSITIVITY = 7.5 # MMseqs2 sensitivity (-s parameter). RCSB default is 7.5
    DB_CLUSTER_ID = 0.95     # Identity for PDB database clustering
    DB_CLUSTER_COV = 0.8     # Coverage for PDB database clustering (0.8 is RCSB standard)

    # GENERATE DESCRIPTIVE RUN ID
    # Shorten settings for filename
    q_short = "all" if QUERY_MODE == "all" else "only" if QUERY_MODE == "only_po4" else "no"
    m_short = "Xray" if EXPERIMENTAL_METHOD == "X-RAY DIFFRACTION" else "All"
    
    # Extract E-value exponent (e.g., 1e-16 -> 16)
    import math
    try:
        e_exponent = int(abs(math.log10(EVALUE_CUTOFF)))
    except:
        e_exponent = "var"

    # Construct the descriptive Run ID
    # Format: Mode_QMode_Res_Rfree_Method_EExp_SId_SCov_SSens_CId_CCov
    run_id = (f"{LIGAND_DETECTION_MODE}_{q_short}_{RES_LIMIT}_{RFREE_LIMIT}_{m_short}_"
              f"{e_exponent}_{SEARCH_IDENTITY}_{SEARCH_COV}_{SEARCH_SENSITIVITY}_{DB_CLUSTER_ID}_{DB_CLUSTER_COV}")
    
    if TM_FAST_MODE:
        run_id += "_fast"

    # Initialize PDB Downloader if needed
    downloader = None
    if RUN_DOWNLOAD_STEP:
        from pdb_downloader import PDBDownloader
        # Start with 5 workers to be safe during the background search
        downloader = PDBDownloader(PROJECT_ROOT, max_workers=5)

    # DEFINE OUTPUT FILE PATHS
    ligand_search_file = os.path.join(PROJECT_ROOT, "results", f"ligand_search_{run_id}.json")
    final_output_file = os.path.join(PROJECT_ROOT, "results", f"homology_enrichment_{run_id}.json")
    tmalign_output_file = os.path.join(PROJECT_ROOT, "results", f"tmalign_results_{run_id}.json")

    # CASE 1: BOTH FILES EXIST (Complete skip OR Standalone Download/Align)
    if os.path.exists(final_output_file) and os.path.exists(ligand_search_file):
        print("\n" + "="*80)
        print(f"ANALYSIS COMPLETE")
        print("-" * 80)
        print(f"  Run ID:                {run_id}")
        
        with open(final_output_file, 'r') as f:
            enrichment_results = json.load(f)

        if RUN_DOWNLOAD_STEP or RUN_TMALIGN_STEP:
            run_concurrent_workflow(enrichment_results, downloader, run_id, RUN_DOWNLOAD_STEP, RUN_TMALIGN_STEP)
        
        print("\n" + "="*80 + "\n")
        return

    # CASE 2: LIGAND SEARCH EXISTS, BUT ENRICHMENT DOES NOT (Resuming)
    if os.path.exists(ligand_search_file):
        print("\n" + "="*80)
        print(f"RESUMING FROM CACHED LIGAND SEARCH")
        print("-" * 80)
        print(f"  Found existing ligand results: {os.path.basename(ligand_search_file)}")
        print("  Skipping steps 1-3 and proceeding directly to Step 4: Homology Enrichment.")
        print("="*80 + "\n")
        
        with open(ligand_search_file, 'r') as f:
            cached_data = json.load(f)
            grouped_by_uniprot = cached_data["grouped_by_uniprot"]
            high_quality_phosphate_pdb_codes = cached_data["high_quality_phosphate_pdb_codes"]
        
        # Proceed directly to Step 4
        enrichment_results = run_blast_and_filter(
            grouped_by_uniprot, 
            high_quality_phosphate_pdb_codes,
            evalue=EVALUE_CUTOFF,
            identity=SEARCH_IDENTITY,
            res_limit=RES_LIMIT,
            rfree_limit=RFREE_LIMIT,
            method=EXPERIMENTAL_METHOD,
            similarity_threshold=DB_CLUSTER_ID,
            run_id=run_id,
            sensitivity=SEARCH_SENSITIVITY,
            search_cov=SEARCH_COV,
            db_cluster_cov=DB_CLUSTER_COV,
            downloader=downloader if EARLY_SEED_DOWNLOAD else None
        )
        
        # Step 5 & 6: Concurrent Download and Alignment
        if RUN_DOWNLOAD_STEP or RUN_TMALIGN_STEP:
            run_concurrent_workflow(enrichment_results, downloader, run_id, RUN_DOWNLOAD_STEP, RUN_TMALIGN_STEP)
        return

    # CASE 3: FULL RUN
    # Store parameters for logging
    run_params = {
        "LIGAND_DETECTION_MODE": LIGAND_DETECTION_MODE,
        "QUERY_MODE": QUERY_MODE,
        "RES_LIMIT": RES_LIMIT,
        "RFREE_LIMIT": RFREE_LIMIT,
        "EXPERIMENTAL_METHOD": EXPERIMENTAL_METHOD,
        "EVALUE_CUTOFF": EVALUE_CUTOFF,
        "SEARCH_IDENTITY": SEARCH_IDENTITY,
        "SEARCH_COV": SEARCH_COV,
        "SEARCH_SENSITIVITY": SEARCH_SENSITIVITY,
        "DB_CLUSTER_ID": DB_CLUSTER_ID,
        "DB_CLUSTER_COV": DB_CLUSTER_COV,
        "run_id": run_id
    }

    print("\n" + "="*80)
    print(f"RUN CONFIGURATION (MMseqs2 Offline Mode)")
    print("-" * 80)
    print(f"  Run ID:            {run_id}")
    print(f"  Ligand Detection:  {LIGAND_DETECTION_MODE}")
    print(f"  RCSB Query Mode:   {QUERY_MODE}")
    print(f"  Quality Criteria:  Res <= {RES_LIMIT}, R-free <= {RFREE_LIMIT}, Method: {m_short}")
    print(f"  Search Settings:   E <= {EVALUE_CUTOFF}, ID >= {SEARCH_IDENTITY}, Cov >= {SEARCH_COV}, Sens: {SEARCH_SENSITIVITY}")
    print(f"  DB Cluster:        ID >= {DB_CLUSTER_ID}, Cov >= {DB_CLUSTER_COV}")
    print("="*80 + "\n")

    print(f"Step 1: Identifying phosphate-containing ligands (Mode: {LIGAND_DETECTION_MODE})...")
    
    # Resolve paths to data files
    cif_file_path = os.path.join(PROJECT_ROOT, "data", "components.cif")
    smi_file_path = os.path.join(PROJECT_ROOT, "data", "Components-smiles-stereo-cactvs.smi")
    
    # Get the set of ligands that have a phosphate group based on selected mode
    if LIGAND_DETECTION_MODE == "2D":
        phosphate_ligands = get_phosphate_ligands_2d(smi_file_path, mode=QUERY_MODE)
    elif LIGAND_DETECTION_MODE == "3D":
        phosphate_ligands = get_phosphate_ligands_3d(cif_file_path, mode=QUERY_MODE)
    elif LIGAND_DETECTION_MODE == "2D+3D":
        l_2d = get_phosphate_ligands_2d(smi_file_path, mode=QUERY_MODE)
        l_3d = get_phosphate_ligands_3d(cif_file_path, mode=QUERY_MODE)
        # Combine both sets and remove duplicates (set union)
        phosphate_ligands = l_2d | l_3d
    else:
        print(f"Error: Invalid LIGAND_DETECTION_MODE '{LIGAND_DETECTION_MODE}'. Use '2D', '3D', or '2D+3D'.")
        return
    
    if not phosphate_ligands:
        print("No phosphate-containing ligands found. Exiting.")
        return

    print(f"Found {len(phosphate_ligands)} phosphate-containing ligands.")
    print("-" * 30)
    print(f"Step 2: Querying RCSB for matching PDB structures (Mode: {QUERY_MODE}, Method: {EXPERIMENTAL_METHOD}, Res: {RES_LIMIT})...")

    # Perform the query based on the selected mode
    high_quality_phosphate_pdb_codes = query_rcsb(
        phosphate_ligands, 
        mode=QUERY_MODE, 
        min_res=RES_LIMIT, 
        max_r_free=RFREE_LIMIT, 
        method=EXPERIMENTAL_METHOD
    )

    if not high_quality_phosphate_pdb_codes:
        print("No PDB entries found for the given criteria. Exiting.")
        return

    print(f"Found {len(high_quality_phosphate_pdb_codes)} PDB entries matching the criteria.")
    print("-" * 30)
    print("Step 3: Grouping PDB entries by UniProt ID...")

    # Group the selected PDB codes by their UniProt ID
    grouped_by_uniprot = group_pdbs_by_uniprot_id(high_quality_phosphate_pdb_codes)

    # SAVE LIGAND SEARCH RESULTS (using Run ID)
    ligand_search_file = os.path.join(PROJECT_ROOT, "results", f"ligand_search_{run_id}.json")
    results_data = {
        "run_params": run_params,
        "high_quality_phosphate_pdb_codes": high_quality_phosphate_pdb_codes,
        "grouped_by_uniprot": grouped_by_uniprot
    }
    
    with open(ligand_search_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Ligand search results saved to {ligand_search_file}.")
    
    # Also save as intermediate_data.json for compatibility with older analysis scripts
    inter_path = os.path.join(PROJECT_ROOT, "results", "intermediate_data.json")
    with open(inter_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Baseline data saved to {inter_path} for evaluation scripts.")

    # Descriptions for Major EC Classes
    EC_DESCRIPTIONS = {
        "1": "Oxidoreductases",
        "2": "Transferases",
        "3": "Hydrolases",
        "4": "Lyases",
        "5": "Isomerases",
        "6": "Ligases",
        "7": "Translocators",
        "no_EC": "Non-Enzymatic / Unclassified"
    }

    # Print a concise summary of the grouped data
    if grouped_by_uniprot:
        print("\n" + "="*80)
        print("SUMMARY: PDB STRUCTURES GROUPED BY FUNCTION (EC) AND PROTEIN (UNIPROT)")
        print("-" * 80)

        # Sort the keys so UNMAPPED is at the top for visibility
        sorted_majors = sorted(grouped_by_uniprot.keys(), key=lambda x: (0 if x=="UNMAPPED" else 1, x))

        total_proteins = 0
        total_pdbs = 0

        for major_ec in sorted_majors:
            sub_ecs = grouped_by_uniprot[major_ec]
            
            # Calculate totals for this major class
            class_proteins = 0
            class_pdbs = 0
            for sub_ec_data in sub_ecs.values():
                class_proteins += len(sub_ec_data)
                class_pdbs += sum(len(data["pdbs"]) for data in sub_ec_data.values())
            
            total_proteins += class_proteins
            total_pdbs += class_pdbs

            if major_ec == "UNMAPPED":
                print(f"  >>> UNMAPPED STRUCTURES:     {class_pdbs:4} PDBs")
            else:
                description = EC_DESCRIPTIONS.get(major_ec, "Unknown Class")
                print(f"  >>> CLASS {major_ec} ({description:15}): {class_proteins:4} Proteins, {class_pdbs:4} PDBs (across {len(sub_ecs)} Sub-classes)")

        print("-" * 80)
        print(f"  TOTAL: {total_proteins} Unique Proteins and {total_pdbs} PDB Entries.")
        print(f"  Full details available in: {os.path.basename(ligand_search_file)}")
        print("="*80 + "\n")
    else:
        print("No PDB entries could be grouped.")
        return

    # Step 4: Running Homology enrichment (Offline MMseqs2)
    enrichment_results = run_blast_and_filter(
        grouped_by_uniprot, 
        high_quality_phosphate_pdb_codes,
        evalue=EVALUE_CUTOFF,
        identity=SEARCH_IDENTITY,
        res_limit=RES_LIMIT,
        rfree_limit=RFREE_LIMIT,
        method=EXPERIMENTAL_METHOD,
        similarity_threshold=DB_CLUSTER_ID,
        run_id=run_id,
        sensitivity=SEARCH_SENSITIVITY,
        search_cov=SEARCH_COV,
        db_cluster_cov=DB_CLUSTER_COV,
        downloader=downloader if EARLY_SEED_DOWNLOAD else None
    )

    # Step 5 & 6: Concurrent Download and Alignment
    if RUN_DOWNLOAD_STEP or RUN_TMALIGN_STEP:
        run_concurrent_workflow(enrichment_results, downloader, run_id, RUN_DOWNLOAD_STEP, RUN_TMALIGN_STEP)

    # Step 7: Analysis and Isostere Discovery
    if RUN_TMALIGN_STEP:
        print("\n" + "="*80)
        print("STEP 7: EXTRACTING BINDING SITE MOTIFS")
        print("-" * 80)
        motif_dir = os.path.join(PROJECT_ROOT, "results", "motif_analysis")

        batch_motif_extraction.run_extraction(
            input_file=final_output_file,
            tm_results_file=tmalign_output_file,
            ls_file=ligand_search_file,
        )

        print("\n" + "="*80)
        print("STEP 8: ANALYZING RESULTS FOR ISOSTERES")
        print("-" * 80)
        
        # Hierarchical Isostere Report
        analyze_isosteres_simple.analyze_isosteres_simple(motif_dir, include_modified=INCLUDE_MODIFIED_RESIDUES)

        # PLIF Similarity Analysis
        analyze_plif.run_plif_analysis(motif_dir)

        # Sort hits into apo / holo / plif_confirmed
        print("\nStep 8b: Sorting hits...")
        sort_hits_module.MOTIF_DIR = motif_dir
        sort_hits_module.HITS_DIR  = os.path.join(motif_dir, "hits")
        sort_hits_module.REFS_DIR  = os.path.join(motif_dir, "references")
        sort_hits_module.sort_hits()

        # DataWarrior export
        print("\nStep 8c: Exporting to DataWarrior...")
        smiles_path = os.path.join(PROJECT_ROOT, "data", "Components-smiles-stereo-cactvs.smi")
        dw_output   = os.path.join(motif_dir, "isostere_datawarrior.txt")
        plif_json   = os.path.join(motif_dir, "plif_results.json")
        export_to_datawarrior.export_for_datawarrior(
            plif_json, smiles_path, dw_output,
            confirmed_only=True,
            confirmed_dir=os.path.join(motif_dir, "hits", "holo", "plif_confirmed"),
        )

        print("="*80 + "\n")

def run_concurrent_workflow(enrichment_results, downloader, run_id, run_download, run_tmalign):
    """
    Orchestrates Steps 5 and 6 concurrently.
    Downloads seeds first, then starts aligning hits as they finish downloading.
    """
    from tmalign_handler import TMAlignHandler
    from pdb_downloader import PDBDownloader
    
    if downloader is None:
        downloader = PDBDownloader(PROJECT_ROOT)

    tmalign_output_file = os.path.join(PROJECT_ROOT, "results", f"tmalign_results_{run_id}.json")
    
    # Detect fast mode from run_id
    fast_mode = "_fast" in run_id

    seeds = set(enrichment_results.get("cluster_reps", []))
    hits = set(enrichment_results.get("new_homologs_list", []))
    hits = hits - seeds
    
    print("\n" + "-" * 30)
    print("Step 5 & 6: Concurrent Download and Structural Alignment...")
    print(f"  Target: {len(seeds)} Seeds (References) and {len(hits)} New Homologs.")

    # 1. Download seeds first (blocking, as they are references for TM-align)
    if run_download:
        downloader.max_workers = 10
        downloader.download_batch(seeds, category="references")

    # 2. Initialize TM-align Handler. Deliberately does NOT skip just because
    # tmalign_output_file already exists: set_checkpoint() below loads it and
    # submit_alignment() skips any (hit, seed) pair already present, so a
    # rerun safely resumes/extends an existing (possibly incomplete) file
    # instead of silently treating "file exists" as "fully done".
    if run_tmalign:
        tm_handler = TMAlignHandler(PROJECT_ROOT, fast_mode=fast_mode)
        tm_handler.set_checkpoint(tmalign_output_file)

        # Map each hit to ONE seed (single assignment, reverted from testing
        # every hit against every seed that found it -- see project notes:
        # for isostere-finding specifically, seeds that share hits are close
        # structural relatives to begin with, so fully cross-testing them is
        # ~2x the TM-align workload for mostly-redundant confirmations, not
        # new discoveries). For a hit claimed by multiple seeds, the BEST
        # match wins -- lowest e-value first, ties broken by highest identity
        # (same ranking used everywhere else in this pipeline, e.g.
        # blast_enrichment.py's clean_hits.sort(key=lambda x: (x['evalue'],
        # -x['identity']))) -- not whichever seed happened to be processed
        # last in cluster_hits' iteration order (the original behavior: pure
        # accident of JSON key order, unrelated to match quality -- confirmed
        # on this project's own data, 69% of contested hits were won by a
        # seed that was NOT the best match, in some cases losing a 100%-
        # identity match to a 64%-identity one).
        hit_to_seed = {}
        hit_quality = {}  # hit_pdb -> (evalue, -identity) of the current winner
        for seed, data in enrichment_results["cluster_hits"].items():
            seed_pdb = seed[:4].upper()
            for hit in data["hits"]:
                hit_pdb = hit["pdb_id"].upper()
                quality = (hit.get("evalue", float("inf")), -hit.get("identity", 0.0))
                if hit_pdb not in hit_to_seed or quality < hit_quality[hit_pdb]:
                    hit_to_seed[hit_pdb] = seed_pdb
                    hit_quality[hit_pdb] = quality

        # Safety net: single assignment can leave a seed with ZERO hits at
        # all -- every one of its candidates lost to a sibling seed, not
        # just reduced coverage but literally never tested against anything.
        # For any such starved seed, force in its own single BEST candidate
        # hit even though that hit is already claimed elsewhere, so every
        # seed with at least one real candidate gets at least one alignment
        # of its own. "Best" = lowest e-value first, ties broken by highest
        # sequence identity -- the exact ranking blast_enrichment.py already
        # uses elsewhere in this pipeline to pick a best hit per PDB entry
        # (see clean_hits.sort(key=lambda x: (x['evalue'], -x['identity']))).
        covered_seeds = set(hit_to_seed.values())
        rescue_by_hit = {}
        n_rescued = 0
        for seed, data in enrichment_results["cluster_hits"].items():
            seed_pdb = seed[:4].upper()
            if seed_pdb in covered_seeds or not data["hits"]:
                continue
            best_hit = min(
                data["hits"],
                key=lambda h: (h.get("evalue", float("inf")), -h.get("identity", 0.0)),
            )
            rescue_by_hit.setdefault(best_hit["pdb_id"].upper(), []).append(seed_pdb)
            n_rescued += 1
        if n_rescued:
            print(f"  [TM-align] {n_rescued} seed(s) would otherwise get zero coverage "
                  f"(every candidate claimed by a sibling seed); rescuing each with its own best hit.")

        def on_hit_downloaded(pdb_id):
            pdb_id_u = pdb_id.upper()
            seed_pdb = hit_to_seed.get(pdb_id_u)
            if seed_pdb:
                tm_handler.submit_alignment(pdb_id, seed_pdb)
            for rescue_seed in rescue_by_hit.get(pdb_id_u, []):
                tm_handler.submit_alignment(pdb_id, rescue_seed)

        # 4. Start download and align concurrently
        with tm_handler:
            if run_download:
                downloader.download_batch(hits, category="hits", on_downloaded=on_hit_downloaded)
            else:
                # If download is disabled but TM-align is enabled, just queue everything immediately
                print(f"  [TM-align] Queuing {len(hits)} alignments...")
                for hit_pdb in hits:
                    on_hit_downloaded(hit_pdb)
            
            # 5. Wait for all alignments to finish
            tm_handler.collect_results(wait_for_all=True)
            
            print(f"  [TM-align] Completed! Results saved to {tmalign_output_file}")

    else:
        # Just download if TM-align is disabled or already done
        if run_download:
            downloader.download_batch(hits, category="hits")

if __name__ == "__main__":
    main()
