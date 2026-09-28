import os
import json
import argparse
import sys

# Add 'src' to path so we can import our modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from rcsb_search import filter_by_quality
from blast_enrichment import run_homology_enrichment

def run_blast_and_filter(grouped_data, high_quality_phosphate_pdb_codes, 
                         evalue=1e-16, identity=0.5, 
                         res_limit=2.7, rfree_limit=0.25,
                         method="X-RAY DIFFRACTION",
                         similarity_threshold=0.95,
                         run_id=None,
                         sensitivity=7.5,
                         search_cov=0.0,
                         db_cluster_cov=0.8,
                         downloader=None):
    """
    Performs the sequence-clustered MMseqs2 enrichment and quality filtering.
    """
    print("-" * 30)
    print(f"Running MMseqs2 enrichment (E={evalue}, ID={identity}, DB Cluster={similarity_threshold})...")
    
    # Run homology enrichment
    # returns dict: { original_pdb_id: { "initiating_pdb": ..., "hits": [...] } }
    pdb_to_homologs = run_homology_enrichment(
        grouped_data, 
        evalue=evalue, 
        identity=identity,
        similarity_threshold=similarity_threshold,
        min_res=res_limit,
        max_r_free=rfree_limit,
        method=method,
        run_id=run_id,
        sensitivity=sensitivity,
        search_cov=search_cov,
        db_cluster_cov=db_cluster_cov,
        downloader=downloader
    )
    
    print("-" * 30)
    print(f"Enrichment complete. Found homologs for {len(pdb_to_homologs)} unique PDB structures.")

    # 1. Collect ALL unique PDB IDs found via BLAST and all cluster representatives
    all_blast_pdbs = set()
    all_reps = set()
    for data in pdb_to_homologs.values():
        all_reps.add(data["initiating_pdb"].upper())
        for hit in data["hits"]:
            all_blast_pdbs.add(hit['pdb_id'].upper())

    # 2. Identify truly new high-quality PDBs
    original_pdbs_set = set(code.upper() for code in high_quality_phosphate_pdb_codes)
    new_pdbs_found = all_blast_pdbs - original_pdbs_set
    
    print(f"Total Cluster Representatives (Seeds): {len(all_reps)}")
    print(f"Total NEW high-quality PDB homologs found: {len(new_pdbs_found)}")

    print("\n" + "="*80)
    print("NEWLY DISCOVERED HIGH-QUALITY HOMOLOGS")
    print("="*80)

    # Group new PDBs by the cluster/UniProt that found them
    found_any_new = False
    
    # Group results by the "initiating" PDB of each cluster
    cluster_results = {}
    for orig_pdb, data in pdb_to_homologs.items():
        orig_pdb_upper = orig_pdb.upper()
        rep_pdb = data["initiating_pdb"]
        hits = data["hits"]
        
        if rep_pdb not in cluster_results:
            new_hits = [h for h in hits 
                        if h['pdb_id'].upper() in new_pdbs_found]
            if new_hits:
                cluster_results[rep_pdb] = {
                    "hits": new_hits,
                    "members": []
                }
        
        if rep_pdb in cluster_results:
            cluster_results[rep_pdb]["members"].append(orig_pdb_upper)

    for rep_pdb, data in sorted(cluster_results.items()):
        found_any_new = True
        hit_ids = [h['pdb_id'].upper() for h in data["hits"]]
        members_str = ", ".join(sorted(data["members"]))
        print(f"  Cluster Rep: {rep_pdb:5} | Members: ({members_str})")
        print(f"    --> Found {len(hit_ids):2} NEW homologs: {', '.join(hit_ids)}")
        print("-" * 40)

    if not found_any_new:
        print("No new high-quality homologous structures were found outside the original results.")
    else:
        print(f"\nSummary: Successfully identified {len(new_pdbs_found)} new structures across {len(cluster_results)} unique protein clusters.")

    # 3. SAVE ENRICHED RESULTS
    enriched_results = {
        "parameters": {
            "evalue": evalue,
            "identity": identity,
            "res_limit": res_limit,
            "rfree_limit": rfree_limit,
            "similarity_threshold": similarity_threshold
        },
        "new_homologs_count": len(new_pdbs_found),
        "new_homologs_list": sorted(list(new_pdbs_found)),
        "cluster_hits": cluster_results,
        "cluster_reps": sorted(list(all_reps)),
        "original_pdbs": list(original_pdbs_set)
    }
    
    # Generate filename based on run_id or parameters
    if run_id:
        filename = f"homology_enrichment_{run_id}.json"
    else:
        import math
        try: e_exp = int(abs(math.log10(evalue)))
        except: e_exp = "var"
        filename = f"homology_enrichment_{e_exp}_{identity}_{similarity_threshold}.json"
    
    output_path = os.path.join(PROJECT_ROOT, "results", filename)
    
    with open(output_path, 'w') as f:
        json.dump(enriched_results, f, indent=2)
    
    print(f"Final enriched results saved to: {output_path}")

    # 4. CLEANUP
    from mmseqs_handler import MMseqsHandler
    handler = MMseqsHandler(PROJECT_ROOT)
    handler.clean_up_run(run_id)

    return enriched_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone MMseqs2 homology enrichment and filtering.")
    parser.add_argument("--input", default="ligand_search_results.json", help="Path to ligand search results JSON.")
    parser.add_argument("--evalue", type=float, default=1e-16, help="E-value cutoff.")
    parser.add_argument("--identity", type=float, default=0.5, help="Search Identity cutoff (0.0 to 1.0).")
    parser.add_argument("--cov", type=float, default=0.5, help="Search Coverage cutoff.")
    parser.add_argument("--cluster_id", type=float, default=0.95, help="DB Clustering identity.")
    parser.add_argument("--res", type=float, default=2.7, help="Resolution cutoff.")
    parser.add_argument("--rfree", type=float, default=0.25, help="R-free cutoff.")
    
    args = parser.parse_args()
    
    # Resolve absolute path for the input file
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # Check if input is a simple filename in results/ or a direct path
    if os.path.exists(args.input):
        input_path = args.input
    else:
        input_path = os.path.join(PROJECT_ROOT, "results", args.input)
    
    if os.path.exists(input_path):
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        run_blast_and_filter(
            data["grouped_by_uniprot"], 
            data["high_quality_phosphate_pdb_codes"],
            evalue=args.evalue,
            identity=args.identity,
            res_limit=args.res,
            rfree_limit=args.rfree,
            similarity_threshold=args.cluster_id
        )
    else:
        print(f"Error: Input file {input_path} not found. Run main.py first to generate it.")
