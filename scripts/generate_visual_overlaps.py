
import json
import random
import os
import subprocess
import platform
import glob
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from utils import apply_transformation, to_wsl_path
from utils import find_structure_in_dir as find_file

def get_pdb_lines(pdb_path, chain_id):
    """Reads a PDB and forces a chain ID."""
    lines = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("TER"):
                new_line = list(line)
                if len(new_line) > 21: new_line[21] = chain_id
                lines.append("".join(new_line))
    return lines

def convert_cif_to_pdb_atoms(cif_path, tmalign_exec):
    """Uses TM-align to convert CIF to PDB atoms by aligning it to itself."""
    is_windows = (platform.system() == "Windows")
    temp_prefix = cif_path.replace(".cif", "_conv_tmp")
    
    if is_windows:
        c_path_wsl = to_wsl_path(os.path.abspath(cif_path))
        o_path_wsl = to_wsl_path(os.path.abspath(temp_prefix))
        t_exec_wsl = to_wsl_path(os.path.abspath(tmalign_exec))
        command = ["wsl", t_exec_wsl, c_path_wsl, c_path_wsl, "-o", o_path_wsl]
    else:
        command = [tmalign_exec, cif_path, cif_path, "-o", temp_prefix]
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        atm_file = temp_prefix + "_all_atm_lig"
        if not os.path.exists(atm_file):
            atm_file = temp_prefix + "_all_atm"
            
        # Extract Chain A from the output
        lines = []
        if os.path.exists(atm_file):
            with open(atm_file, 'r') as f:
                for line in f:
                    if (line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("TER")) and " A " in line:
                        lines.append(line)
        
        # Cleanup
        for f in glob.glob(temp_prefix + "*"):
            os.remove(f)
            
        return lines
    except Exception as e:
        print(f"  [Error] CIF conversion failed: {e}")
        return []

def main():
    project_root = r"D:\AI\Master_thesis\Phosphate-binding-site"
    results_file = os.path.join(project_root, "results", "tmalign_results_3D_no_2.7_0.25_Xray_16_0.5_0.0_7.5_0.95_0.8_fast.json")
    tmalign_exec = os.path.join(project_root, "scripts", "Shell scripts", "TMalign_linux")
    ref_dir = os.path.join(project_root, "data", "structures", "references")
    hits_dir = os.path.join(project_root, "data", "structures", "hits")
    output_dir = os.path.join(project_root, "results")

    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    all_pairs = []
    for seed, hits in data.items():
        for hit in hits:
            all_pairs.append({
                "seed": seed,
                "hit": hit["hit"],
                "tm_score": hit["tm_score_1"],
                "matrix": hit["transformation"]
            })

    all_pairs.sort(key=lambda x: x["tm_score"])

    # Sample selection
    high_hits = [p for p in all_pairs if p["tm_score"] > 0.8]
    med_hits = [p for p in all_pairs if 0.4 <= p["tm_score"] <= 0.6]
    low_hits = [p for p in all_pairs if p["tm_score"] < 0.3]

    selected = {}
    if high_hits: selected["high"] = random.choice(high_hits)
    if med_hits: selected["medium"] = random.choice(med_hits)
    if low_hits: selected["low"] = random.choice(low_hits)

    for level, pair in selected.items():
        seed_id = pair['seed']
        hit_id = pair['hit']
        matrix = pair['matrix']
        
        print(f"\nProcessing {level} TM-score sample: {seed_id} vs {hit_id} (TM={pair['tm_score']:.3f})")
        
        hit_path = find_file(hit_id, hits_dir)
        seed_path = find_file(seed_id, ref_dir)
        
        if not hit_path or not seed_path:
            print(f"  [Error] Missing files for {seed_id} or {hit_id}")
            continue

        # 1. Transform Hit (Chain A)
        # Hit is always PDB in this project based on directory check
        print(f"  Applying transformation matrix to {hit_id}...")
        chain_a_lines = apply_transformation(hit_path, matrix, "A")
        
        # 2. Get Seed (Chain B)
        print(f"  Preparing reference structure {seed_id}...")
        if seed_path.endswith(".cif"):
            chain_b_lines = convert_cif_to_pdb_atoms(seed_path, tmalign_exec)
            # Force chain B
            chain_b_lines = [l[:21] + "B" + l[22:] if len(l) > 21 else l for l in chain_b_lines]
        else:
            chain_b_lines = get_pdb_lines(seed_path, "B")
            
        # 3. Write Overlap PDB
        output_filename = f"visual_overlap_{level}_{seed_id}_vs_{hit_id}.pdb"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as f:
            f.write(f"REMARK 000 Visual overlap {level} TM-score\n")
            f.write(f"REMARK 000 Query (Chain B): {seed_id}\n")
            f.write(f"REMARK 000 Hit (Chain A): {hit_id} (Transformed)\n")
            f.write(f"REMARK 000 TM-score: {pair['tm_score']:.4f}\n")
            f.writelines(chain_a_lines)
            f.write("TER\n")
            f.writelines(chain_b_lines)
            f.write("END\n")
            
        print(f"  [Success] Saved to {output_path}")

if __name__ == "__main__":
    main()
