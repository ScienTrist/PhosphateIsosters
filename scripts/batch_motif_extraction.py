
import os
import json
import random
import gemmi
import sys
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from constants import STANDARD_AA, EC_NAMES

def get_sequence_identity(seq1, seq2):
    """
    Calculates sequence identity based on residue numbers (dicts of num -> name).
    Robust against shifts and missing residues.
    """
    if not seq1 or not seq2: return 0.0
    all_nums = set(seq1.keys()) | set(seq2.keys())
    matches = sum(1 for n in all_nums if n in seq1 and n in seq2 and seq1[n] == seq2[n])
    return matches / len(all_nums)

def find_structure_file(pdb_id, project_root):
    for d in ["hits", "references"]:
        for ext in [".pdb", ".cif"]:
            path = os.path.join(project_root, "data", "structures", d, f"{pdb_id.upper()}{ext}")
            if os.path.exists(path): return path
    return None

def get_unique_ref_sites(structure):
    """
    Identifies unique phosphate binding sites in a reference structure.
    1. Clusters chains by 95% seq identity.
    2. Only processes the FIRST chain found for each cluster (Subunit Deduplication).
    3. Spatially clusters sites within 5.0A (P-P distance) ONLY if they have the SAME ligand name.
    """
    # 1. Map chains to their sequence (num -> name)
    chain_seqs = {} # chain_name -> {res_num: res_name}
    for model in structure:
        for chain in model:
            res_dict = {r.seqid.num: r.name for r in chain if r.name.upper() in STANDARD_AA}
            if res_dict:
                chain_seqs[chain.name] = res_dict

    # 2. Cluster chains by 95% identity
    chain_to_cluster = {}
    cluster_representatives = {} # cluster_id -> first_chain_name
    sorted_chains = sorted(list(chain_seqs.keys()))
    for i, c1 in enumerate(sorted_chains):
        if c1 in chain_to_cluster: continue
        
        cluster_id = c1
        chain_to_cluster[c1] = cluster_id
        cluster_representatives[cluster_id] = c1 # First one is the rep
        
        for j in range(i + 1, len(sorted_chains)):
            c2 = sorted_chains[j]
            if get_sequence_identity(chain_seqs[c1], chain_seqs[c2]) >= 0.95:
                chain_to_cluster[c2] = cluster_id

    # 3. Collect all phosphate residues ONLY from the representative chains
    # This fulfills: "if a chain is 0.95 identical... only 1 is extracted"
    rep_chain_names = set(cluster_representatives.values())
    sites_by_name = defaultdict(list) # (lig_name, cluster_id) -> list of site data
    
    for model in structure:
        for chain in model:
            if chain.name not in rep_chain_names:
                continue
            
            cluster_id = chain_to_cluster[chain.name]
            for res in chain:
                # Explicitly skip simple phosphate ions and water
                if res.name in ["PO4", "PI", "2HP", "HOH"]:
                    continue
                    
                p_atoms = [a for a in res if a.element.name == "P"]
                if p_atoms:
                    key = (res.name, cluster_id)
                    sites_by_name[key].append({
                        'res': res, 
                        'chain': chain.name, 
                        'p_pos': [a.pos for a in p_atoms]
                    })

    # 4. Spatially cluster sites ONLY within the same (name, cluster_id) group
    # This fulfills: "supersite logic... should only group on the same ligand"
    final_site_clusters = []
    
    for (lig_name, cluster_id), sites in sites_by_name.items():
        n = len(sites)
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                min_pp_dist = min((p1.dist(p2) for p1 in sites[i]['p_pos'] for p2 in sites[j]['p_pos']), default=999.0)
                if min_pp_dist <= 5.0:
                    adj[i].append(j)
                    adj[j].append(i)
        
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                component = []
                stack = [i]
                visited[i] = True
                while stack:
                    curr = stack.pop()
                    component.append((sites[curr]['res'], sites[curr]['chain']))
                    for neighbor in adj[curr]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)
                final_site_clusters.append(component)
    
    return final_site_clusters

def extract_and_save_split(st_ref, st_hit, ref_id, hit_id, ref_sites, output_dirs, existing_files):
    """
    Extracts the environment for each site in ref_sites.
    Robustly handles hits by using spatial proximity instead of residue matching.
    """
    extracted_count = 0
    radius = 10.0

    # Helper to map chain names to single characters for PDB format compatibility
    def get_safe_chain_name(name, mapping, used_chars):
        if name in mapping: return mapping[name]
        preferred = name[0] if name else 'A'
        if preferred not in used_chars:
            mapping[name] = preferred
            used_chars.add(preferred)
            return preferred
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz":
            if char not in used_chars:
                mapping[name] = char
                used_chars.add(char)
                return char
        return "?"

    for cluster_ligands in ref_sites:
        # Use the first ligand in the cluster as the spatial anchor
        best_lig, best_chain = cluster_ligands[0]
        
        lig_pos = [a.pos for a in best_lig]
        # Filenames are based on the Reference site info
        ref_base = f"ref_{ref_id}_{best_chain}_{best_lig.name}_{best_lig.seqid.num}.cif"
        hit_base = f"hit_{ref_id}_{hit_id}_{best_chain}_{best_lig.name}_{best_lig.seqid.num}.cif"
        
        # 1. Skip if hit already exists
        if hit_base in existing_files:
            extracted_count += 1
            continue

        ref_out_st = gemmi.Structure(); ref_model = gemmi.Model("1")
        hit_out_st = gemmi.Structure(); hit_model = gemmi.Model("1")
        
        # 2. Extract from Reference (if missing)
        ref_p = os.path.join(output_dirs['ref'], ref_base)
        if not os.path.exists(ref_p):
            ref_name_map, ref_used = {}, set()
            for model in st_ref:
                for chain in model:
                    indices = [i for i, r in enumerate(chain) if any(any(a.pos.dist(lp) <= radius for lp in lig_pos) for a in r)]
                    if not indices: continue
                    safe_name = get_safe_chain_name(chain.name.lower(), ref_name_map, ref_used)
                    segments = []
                    curr = [indices[0]]
                    for idx in indices[1:]:
                        if idx == curr[-1] + 1: curr.append(idx)
                        else: segments.append(curr); curr = [idx]
                    segments.append(curr)
                    for seg in segments:
                        new_c = gemmi.Chain(safe_name)
                        for idx in seg: new_c.add_residue(chain[idx].clone(), -1)
                        ref_model.add_chain(new_c)
            if len(ref_model) > 0:
                ref_out_st.add_model(ref_model)
                ref_out_st.setup_entities()
                ref_out_st.assign_label_seq_id()
                ref_out_st.make_mmcif_document().write_file(ref_p)

        # 3. Extract from Hit (Spatial proximity is robust against numbering)
        hit_p = os.path.join(output_dirs['hit'], hit_base)
        hit_name_map, hit_used = {}, set()
        for model in st_hit:
            for chain in model:
                indices = [i for i, r in enumerate(chain) if any(any(a.pos.dist(lp) <= radius for lp in lig_pos) for a in r)]
                if not indices: continue
                safe_name = get_safe_chain_name(chain.name.upper(), hit_name_map, hit_used)
                segments = []
                curr = [indices[0]]
                for idx in indices[1:]:
                    if idx == curr[-1] + 1: curr.append(idx)
                    else: segments.append(curr); curr = [idx]
                segments.append(curr)
                for seg in segments:
                    new_c = gemmi.Chain(safe_name)
                    for idx in seg: new_c.add_residue(chain[idx].clone(), -1)
                    hit_model.add_chain(new_c)
        
        if len(hit_model) > 0:
            hit_out_st.add_model(hit_model)
            hit_out_st.setup_entities()
            hit_out_st.assign_label_seq_id()
            hit_out_st.make_mmcif_document().write_file(hit_p)
            extracted_count += 1
            
    return extracted_count

def process_single_reference(ref_id, hits, ec_map, EC_NAMES, PROJECT_ROOT, existing_files):
    """
    Worker function to process all hits for a single reference structure.
    """
    try:
        base_out = os.path.join(PROJECT_ROOT, "results", "motif_analysis")
        dirs = {'ref': os.path.join(base_out, "references"), 'hit': os.path.join(base_out, "hits")}
        for d in dirs.values(): os.makedirs(d, exist_ok=True)

        ref_path = find_structure_file(ref_id, PROJECT_ROOT)
        if not ref_path: return 0, 0

        st_ref = gemmi.read_structure(ref_path)
        ref_sites = get_unique_ref_sites(st_ref)
        
        if not ref_sites: return 0, 0

        local_total_sites = 0
        local_pairs = 0
        for hit_data in hits:
            hit_id = hit_data['hit']
            hit_path = find_structure_file(hit_id, PROJECT_ROOT)
            if not hit_path: continue

            st_hit = gemmi.read_structure(hit_path)
            # Transform Hit
            t, u = hit_data['transformation']['t'], hit_data['transformation']['u']
            for model in st_hit:
                for chain in model:
                    for res in chain:
                        for atom in res:
                            p = atom.pos
                            nx = t[0] + u[0][0]*p.x + u[0][1]*p.y + u[0][2]*p.z
                            ny = t[1] + u[1][0]*p.x + u[1][1]*p.y + u[1][2]*p.z
                            nz = t[2] + u[2][0]*p.x + u[2][1]*p.y + u[2][2]*p.z
                            atom.pos = gemmi.Position(nx, ny, nz)
            
            sites = extract_and_save_split(st_ref, st_hit, ref_id, hit_id, ref_sites, dirs, existing_files)
            local_total_sites += sites
            local_pairs += 1
            
        return local_total_sites, local_pairs
    except Exception as e:
        print(f"Error processing {ref_id}: {e}")
        return 0, 0

def run_extraction(input_file, tm_results_file, ls_file, workers=8, ec_filter=None, limit=0):
    """
    Core extraction logic — callable from main.py or standalone.
    All file paths are passed explicitly so main.py can supply dynamic run_id paths.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found."); return
    if not os.path.exists(tm_results_file) or os.path.getsize(tm_results_file) <= 2:
        print(f"Error: {tm_results_file} not found or empty."); return

    with open(tm_results_file, 'r') as f:
        tm_data = json.load(f)

    # Pre-scan existing files for idempotency
    print("Pre-scanning existing files for faster idempotency checks...")
    existing_files = set()
    motif_path = os.path.join(PROJECT_ROOT, "results", "motif_analysis")
    if os.path.exists(motif_path):
        for root, _, files in os.walk(motif_path):
            existing_files.update(files)
    print(f"  Found {len(existing_files)} existing motif files.")

    # EC mapping from ligand search results
    ec_map = {}
    if ls_file and os.path.exists(ls_file):
        with open(ls_file, 'r') as f:
            ls_data = json.load(f)
            for major_ec, subs in ls_data.get('grouped_by_uniprot', {}).items():
                for sub_ec, up_ids in subs.items():
                    for up_id, up_data in up_ids.items():
                        for pdb in up_data.get('pdbs', []):
                            ec_map[pdb.upper()] = major_ec

    ref_to_hits = {}
    for ref_id, hits in tm_data.items():
        if ec_filter and ec_map.get(ref_id.upper()) != str(ec_filter):
            continue
        valid_hits = [h for h in hits if h['tm_score_1'] > 0.7]
        if valid_hits:
            ref_to_hits[ref_id] = valid_hits

    target_refs = sorted(ref_to_hits.keys())
    if limit > 0:
        target_refs = target_refs[:limit]

    total_pairs_to_process = sum(len(ref_to_hits[r]) for r in target_refs)
    print(f"Starting PARALLEL batch extraction with {workers} workers...")
    print(f"Found {len(target_refs)} reference structures with valid hits.")
    print(f"Total pairs to process: {total_pairs_to_process}")

    total_sites = total_pairs_processed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_single_reference, rid, ref_to_hits[rid], ec_map, EC_NAMES, PROJECT_ROOT, existing_files): rid
            for rid in target_refs
        }
        for i, future in enumerate(as_completed(futures)):
            ref_id = futures[future]
            try:
                sites, pairs = future.result()
                total_sites += sites
                total_pairs_processed += pairs
                if (i + 1) % 10 == 0 or i == len(target_refs) - 1:
                    print(f"  [{i+1}/{len(target_refs)}] Finished {ref_id}. Cumulative sites: {total_sites}")
            except Exception as e:
                print(f"  Error in future for {ref_id}: {e}")

    print(f"\nDone! Total pairs processed: {total_pairs_processed}/{total_pairs_to_process}")
    print(f"Total binding sites extracted: {total_sites}")


def main():
    parser = argparse.ArgumentParser(description="Batch extract non-redundant binding site motifs.")
    parser.add_argument("--ec", help="Filter by Enzyme Class major digit (e.g. 1)")
    parser.add_argument("--input", help="Path to homology enrichment JSON")
    parser.add_argument("--tm", help="Path to TM-align results JSON")
    parser.add_argument("--ls", help="Path to ligand search JSON (for EC mapping)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of refs (0=all)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    default_id = "3D_no_2.7_0.25_Xray_16_0.5_0.8_7.5_0.95_0.8"
    results_dir = os.path.join(PROJECT_ROOT, "results")

    input_file      = args.input or os.path.join(results_dir, f"homology_enrichment_{default_id}.json")
    tm_results_file = args.tm    or os.path.join(results_dir, f"tmalign_results_{default_id}.json")
    ls_file         = args.ls    or os.path.join(results_dir, f"ligand_search_{default_id}.json")

    run_extraction(input_file, tm_results_file, ls_file,
                   workers=args.workers, ec_filter=args.ec, limit=args.limit)


if __name__ == "__main__":
    main()
