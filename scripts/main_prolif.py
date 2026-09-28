# main_prolif.py
# Single entry point for the full ProLIF-based pipeline, one command start to
# finish: Steps 1-6 identical to main.py (ligand detection through TM-align,
# reused by importing run_concurrent_workflow rather than reimplementing it),
# then Step 7 -- ProLIF-native candidate-mimic discovery and interaction
# scoring (ProLIF_v2/scripts/discover_candidates.py, in place of the homebrew
# batch_motif_extraction.py + analyze_plif.py scorer; this is also where every
# candidate's protein/ligand gets protonated, via protonate_protein.py/
# protonate_ligand.py -- the latter includes physiological-pH ligand
# ionization via Dimorphite-DL, not a separate pre-pass), then Step 8 --
# isosteric functional-group clustering (ProLIF_v2/scripts/
# cluster_isosteric_motifs.py, the discovery-run equivalent of the old,
# deprecated build_ifg_prolif_dataset.py/summarize_ifg_groups.py CSV chain).
#
# Config below is kept identical in spelling/formula to main.py's own config
# block so the two scripts derive the SAME run_id for the same settings --
# either script can find and reuse results/{ligand_search,homology_enrichment,
# tmalign_results}_{run_id}.json already produced by the other.

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(SCRIPT_DIR)  # so `from main import run_concurrent_workflow` resolves
sys.path.append(os.path.join(PROJECT_ROOT, "ProLIF_v2", "scripts"))

import json

from ligand_detection_2D import get_phosphate_ligands_2d
from ligand_detection_3D import get_phosphate_ligands_3d
from rcsb_search import query_rcsb
from uniprot_mapping import group_pdbs_by_uniprot_id
from blast_search_standalone import run_blast_and_filter
from main import run_concurrent_workflow  # reuses steps 5-6 AND the seed/hit pairing logic verbatim

import discover_candidates
import cluster_isosteric_motifs
import build_run_report


def resolve_tmalign_json(run_id):
    """Prefers a from-scratch clean regeneration over the plain (older,
    pre-single-assignment-pairing) file -- never a stale/archived _corrected
    file. See results/archived_tmalign_results/README.md for why."""
    results_dir = os.path.join(PROJECT_ROOT, "results")
    clean = os.path.join(results_dir, f"tmalign_results_{run_id}_clean.json")
    plain = os.path.join(results_dir, f"tmalign_results_{run_id}.json")
    if os.path.exists(clean):
        return clean
    if os.path.exists(plain):
        return plain
    raise FileNotFoundError(f"No TM-align results for run_id={run_id} (checked {clean}, {plain})")


def run_prolif_discovery(run_id, tm_score_threshold, min_ref_bits,
                          run_ifg_clustering=True, ifg_min_score=0.0, run_report=True):
    print("\n" + "=" * 80)
    print("STEP 7 (ProLIF): CANDIDATE MIMIC DISCOVERY + INTERACTION SCORING")
    print("-" * 80)
    tmalign_json_path = resolve_tmalign_json(run_id)
    print(f"  Using TM-align results: {os.path.basename(tmalign_json_path)}")
    if min_ref_bits > 0:
        print(f"  Reference-site pre-filter: > {min_ref_bits} non-VdW interaction bits required")
    # Protein/ligand protonation (incl. physiological-pH ligand ionization)
    # happens inside here, per candidate, via discover_candidates.py's own
    # imports of protonate_protein.py/protonate_ligand.py -- not a separate
    # step that needs calling out here.
    discover_candidates.run_discovery(run_id, tmalign_json_path,
                                       min_tm_score=tm_score_threshold, min_ref_bits=min_ref_bits)
    print("=" * 80 + "\n")

    if run_ifg_clustering:
        print("\n" + "=" * 80)
        print("STEP 8 (IFG): ISOSTERIC FUNCTIONAL-GROUP CLUSTERING")
        print("-" * 80)
        # Reuses the tmalign_json_path this same call already resolved above
        # -- no reason to re-resolve it a second time for the same run_id.
        cluster_isosteric_motifs.run_clustering(run_id, min_score=ifg_min_score,
                                                 tmalign_json_path=tmalign_json_path)
        print("=" * 80 + "\n")

    if run_report:
        print("\n" + "=" * 80)
        print("STEP 9: PER-STAGE RUN REPORT")
        print("-" * 80)
        # Reads back exactly what Step 7/8 just wrote (discovery_run_summary_
        # {run_id}.json, isosteric_motif_clusters_{run_id}.json, and both
        # steps' failure CSVs) -- does not touch the pipeline itself, safe to
        # run even if Step 8 was skipped (it just reports Step 7 alone).
        build_run_report.build_report(run_id)
        print("=" * 80 + "\n")


def main():
    """
    Orchestrates the ProLIF-based workflow:
    1. Identifies phosphate-containing ligands (2D, 3D, or Combined).
    2. Queries the RCSB database for PDB structures containing these ligands.
    3. Groups the resulting PDB IDs by their corresponding UniProt ID.
    4. Saves intermediate results to JSON.
    5. Runs BLAST enrichment for homologs with user-defined settings.
    6. Downloads PDB/CIF structure files for analysis.
    7. Performs structural alignment via TM-align.
    8. Discovers and scores candidate phosphate mimics natively through ProLIF
       (protonating each candidate's protein/ligand, with physiological-pH
       ligand ionization, as part of this same step).
    9. Clusters the isosteric functional groups found across the winning
       assignments (IFG step).
    10. Writes a per-stage run report: how much data each step processed,
        failure rate, and top error reasons (Step 9, reads back what Steps
        7/8 already logged -- see build_run_report.py).
    """
    # CONFIGURATION -- kept identical to main.py's own config block/run_id formula.
    LIGAND_DETECTION_MODE = "3D"

    RUN_DOWNLOAD_STEP = True
    EARLY_SEED_DOWNLOAD = True

    # False by default: run_concurrent_workflow's own TMAlignHandler checkpoints
    # against the PLAIN tmalign_results_{run_id}.json, not the _clean.json file
    # this script's own discovery step resolves and uses (see resolve_tmalign_json).
    # That plain file was moved to results/archived_tmalign_results/ -- if this
    # were True with nothing at that plain path, TMAlignHandler would silently
    # redo the entire ~4-hour alignment from scratch into a new plain file
    # before discovery even starts, purely redundant since alignment is already
    # done and sitting correctly in the clean file. Only set this True if you
    # specifically need to (re)compute alignments this run hasn't produced yet.
    RUN_TMALIGN_STEP = False
    TM_FAST_MODE = False

    QUERY_MODE = "no_po4"

    RES_LIMIT = 2.7
    RFREE_LIMIT = 0.25
    EXPERIMENTAL_METHOD = "X-RAY DIFFRACTION"

    EVALUE_CUTOFF = 1e-16
    SEARCH_IDENTITY = 0.5
    SEARCH_COV = 0.8
    SEARCH_SENSITIVITY = 7.5
    DB_CLUSTER_ID = 0.95
    DB_CLUSTER_COV = 0.8

    # PROLIF DISCOVERY SETTINGS
    RUN_PROLIF_DISCOVERY = True
    TM_SCORE_THRESHOLD = 0.7  # only hits with tm_score_1 >= this are candidates for discovery
    # Skips entire reference sites whose busiest phosphate group has <= this many
    # non-VdW interaction bits, BEFORE protonating any of their hits -- same
    # filter/threshold build_ifg_prolif_dataset.py's --min-ref-bits used
    # (minrefbits1 in that script's earlier output filenames). A reference site
    # with too little real chemical signal can never produce a meaningful score
    # against any hit, so this skips the (much larger) hit-side protonation +
    # extraction + fingerprinting cost for sites that were never going to be
    # useful. Set to 0 to process every reference site (original, unfiltered
    # behavior).
    MIN_REF_BITS = 1

    # IFG CLUSTERING SETTINGS (Step 8, runs immediately after Step 7 using
    # that same run's fresh prolif_discovery_results_{run_id}.json -- see
    # cluster_isosteric_motifs.py)
    RUN_IFG_CLUSTERING = True
    IFG_MIN_SCORE = 0.0  # only assignments with prolif_plif_score >= this are clustered

    # RUN REPORT SETTINGS (Step 9, final -- data volume/failure-rate/error-
    # reason summary across Steps 7+8, see build_run_report.py)
    RUN_FINAL_REPORT = True

    q_short = "all" if QUERY_MODE == "all" else "only" if QUERY_MODE == "only_po4" else "no"
    m_short = "Xray" if EXPERIMENTAL_METHOD == "X-RAY DIFFRACTION" else "All"

    import math
    try:
        e_exponent = int(abs(math.log10(EVALUE_CUTOFF)))
    except Exception:
        e_exponent = "var"

    run_id = (f"{LIGAND_DETECTION_MODE}_{q_short}_{RES_LIMIT}_{RFREE_LIMIT}_{m_short}_"
              f"{e_exponent}_{SEARCH_IDENTITY}_{SEARCH_COV}_{SEARCH_SENSITIVITY}_{DB_CLUSTER_ID}_{DB_CLUSTER_COV}")

    if TM_FAST_MODE:
        run_id += "_fast"

    downloader = None
    if RUN_DOWNLOAD_STEP:
        from pdb_downloader import PDBDownloader
        downloader = PDBDownloader(PROJECT_ROOT, max_workers=5)

    ligand_search_file = os.path.join(PROJECT_ROOT, "results", f"ligand_search_{run_id}.json")
    final_output_file = os.path.join(PROJECT_ROOT, "results", f"homology_enrichment_{run_id}.json")

    # CASE 1: BOTH FILES EXIST -- steps 1-4 already done.
    if os.path.exists(final_output_file) and os.path.exists(ligand_search_file):
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE (Steps 1-4)")
        print("-" * 80)
        print(f"  Run ID: {run_id}")

        with open(final_output_file, 'r') as f:
            enrichment_results = json.load(f)

        if RUN_DOWNLOAD_STEP or RUN_TMALIGN_STEP:
            run_concurrent_workflow(enrichment_results, downloader, run_id, RUN_DOWNLOAD_STEP, RUN_TMALIGN_STEP)

        print("\n" + "=" * 80 + "\n")

        if RUN_PROLIF_DISCOVERY:
            run_prolif_discovery(run_id, TM_SCORE_THRESHOLD, MIN_REF_BITS,
                                 RUN_IFG_CLUSTERING, IFG_MIN_SCORE, RUN_FINAL_REPORT)
        return

    # CASE 2: LIGAND SEARCH EXISTS, ENRICHMENT DOES NOT -- resume from Step 4.
    if os.path.exists(ligand_search_file):
        print("\n" + "=" * 80)
        print("RESUMING FROM CACHED LIGAND SEARCH")
        print("-" * 80)
        print(f"  Found existing ligand results: {os.path.basename(ligand_search_file)}")
        print("  Skipping steps 1-3 and proceeding directly to Step 4: Homology Enrichment.")
        print("=" * 80 + "\n")

        with open(ligand_search_file, 'r') as f:
            cached_data = json.load(f)
            grouped_by_uniprot = cached_data["grouped_by_uniprot"]
            high_quality_phosphate_pdb_codes = cached_data["high_quality_phosphate_pdb_codes"]

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
            downloader=downloader if EARLY_SEED_DOWNLOAD else None,
        )

        if RUN_DOWNLOAD_STEP or RUN_TMALIGN_STEP:
            run_concurrent_workflow(enrichment_results, downloader, run_id, RUN_DOWNLOAD_STEP, RUN_TMALIGN_STEP)

        if RUN_PROLIF_DISCOVERY:
            run_prolif_discovery(run_id, TM_SCORE_THRESHOLD, MIN_REF_BITS,
                                 RUN_IFG_CLUSTERING, IFG_MIN_SCORE, RUN_FINAL_REPORT)
        return

    # CASE 3: FULL RUN
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
        "run_id": run_id,
    }

    print("\n" + "=" * 80)
    print("RUN CONFIGURATION (ProLIF pipeline)")
    print("-" * 80)
    print(f"  Run ID:            {run_id}")
    print(f"  Ligand Detection:  {LIGAND_DETECTION_MODE}")
    print(f"  RCSB Query Mode:   {QUERY_MODE}")
    print(f"  Quality Criteria:  Res <= {RES_LIMIT}, R-free <= {RFREE_LIMIT}, Method: {m_short}")
    print(f"  Search Settings:   E <= {EVALUE_CUTOFF}, ID >= {SEARCH_IDENTITY}, Cov >= {SEARCH_COV}, Sens: {SEARCH_SENSITIVITY}")
    print(f"  DB Cluster:        ID >= {DB_CLUSTER_ID}, Cov >= {DB_CLUSTER_COV}")
    print(f"  ProLIF Discovery:  TM-score threshold >= {TM_SCORE_THRESHOLD}, min ref bits > {MIN_REF_BITS}")
    print("=" * 80 + "\n")

    print(f"Step 1: Identifying phosphate-containing ligands (Mode: {LIGAND_DETECTION_MODE})...")

    cif_file_path = os.path.join(PROJECT_ROOT, "data", "components.cif")
    smi_file_path = os.path.join(PROJECT_ROOT, "data", "Components-smiles-stereo-cactvs.smi")

    if LIGAND_DETECTION_MODE == "2D":
        phosphate_ligands = get_phosphate_ligands_2d(smi_file_path, mode=QUERY_MODE)
    elif LIGAND_DETECTION_MODE == "3D":
        phosphate_ligands = get_phosphate_ligands_3d(cif_file_path, mode=QUERY_MODE)
    elif LIGAND_DETECTION_MODE == "2D+3D":
        l_2d = get_phosphate_ligands_2d(smi_file_path, mode=QUERY_MODE)
        l_3d = get_phosphate_ligands_3d(cif_file_path, mode=QUERY_MODE)
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

    high_quality_phosphate_pdb_codes = query_rcsb(
        phosphate_ligands,
        mode=QUERY_MODE,
        min_res=RES_LIMIT,
        max_r_free=RFREE_LIMIT,
        method=EXPERIMENTAL_METHOD,
    )

    if not high_quality_phosphate_pdb_codes:
        print("No PDB entries found for the given criteria. Exiting.")
        return

    print(f"Found {len(high_quality_phosphate_pdb_codes)} PDB entries matching the criteria.")
    print("-" * 30)
    print("Step 3: Grouping PDB entries by UniProt ID...")

    grouped_by_uniprot = group_pdbs_by_uniprot_id(high_quality_phosphate_pdb_codes)

    results_data = {
        "run_params": run_params,
        "high_quality_phosphate_pdb_codes": high_quality_phosphate_pdb_codes,
        "grouped_by_uniprot": grouped_by_uniprot,
    }

    with open(ligand_search_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Ligand search results saved to {ligand_search_file}.")

    if not grouped_by_uniprot:
        print("No PDB entries could be grouped.")
        return

    print("-" * 30)
    print("Step 4: Running Homology enrichment (Offline MMseqs2)...")
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
        downloader=downloader if EARLY_SEED_DOWNLOAD else None,
    )

    if RUN_DOWNLOAD_STEP or RUN_TMALIGN_STEP:
        run_concurrent_workflow(enrichment_results, downloader, run_id, RUN_DOWNLOAD_STEP, RUN_TMALIGN_STEP)

    if RUN_PROLIF_DISCOVERY:
        run_prolif_discovery(run_id, TM_SCORE_THRESHOLD, MIN_REF_BITS,
                                 RUN_IFG_CLUSTERING, IFG_MIN_SCORE, RUN_FINAL_REPORT)


if __name__ == "__main__":
    main()
