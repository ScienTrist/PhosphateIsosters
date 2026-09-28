
import os
import json
import sys
import argparse
import subprocess
import platform
import shutil
import glob

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from tmalign_wrapper import run_tmalign
from utils import apply_transformation, to_wsl_path
from utils import find_structure_in_dir as find_structure_file

def get_pdb_atoms_from_cif(cif_path, tmalign_exec):
    """Uses TM-align to convert CIF to PDB atoms by aligning it to itself."""
    temp_prefix = cif_path.replace(".cif", "_conv_tmp")
    is_windows = (platform.system() == "Windows")
    
    if is_windows:
        c_path_wsl = to_wsl_path(os.path.abspath(cif_path))
        o_path_wsl = to_wsl_path(os.path.abspath(temp_prefix))
        t_exec_wsl = to_wsl_path(os.path.abspath(tmalign_exec))
        command = ["wsl", t_exec_wsl, c_path_wsl, c_path_wsl, "-o", o_path_wsl]
    else:
        command = [tmalign_exec, cif_path, cif_path, "-o", temp_prefix]
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        # We want the all-atom-ligand file
        atm_file = temp_prefix + "_all_atm_lig"
        if not os.path.exists(atm_file):
            atm_file = temp_prefix + "_all_atm"
            
        lines = []
        if os.path.exists(atm_file):
            with open(atm_file, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("TER"):
                        lines.append(line)
        
        # Cleanup
        for f in glob.glob(temp_prefix + "*"):
            try: os.remove(f)
            except: pass
            
        return lines
    except Exception as e:
        print(f"Error converting CIF: {e}")
        return []

def transform_lines(lines, matrix, chain_id):
    """Applies transformation to a list of PDB lines."""
    t = matrix['t']
    u = matrix['u']
    transformed = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                nx = t[0] + u[0][0]*x + u[0][1]*y + u[0][2]*z
                ny = t[1] + u[1][0]*x + u[1][1]*y + u[1][2]*z
                nz = t[2] + u[2][0]*x + u[2][1]*y + u[2][2]*z
                new_line = list(line)
                coord_str = f"{nx:8.3f}{ny:8.3f}{nz:8.3f}"
                new_line[30:54] = list(coord_str)
                if len(new_line) > 21: new_line[21] = chain_id
                transformed.append("".join(new_line))
            except:
                transformed.append(line)
        elif line.startswith("TER"):
            new_line = list(line)
            if len(new_line) > 21: new_line[21] = chain_id
            transformed.append("".join(new_line))
    return transformed

def get_static_lines(pdb_path, chain_id, tmalign_exec):
    """Gets PDB lines for the reference structure, converting from CIF if needed."""
    if pdb_path.endswith(".cif"):
        lines = get_pdb_atoms_from_cif(pdb_path, tmalign_exec)
    else:
        with open(pdb_path, 'r') as f:
            lines = [l for l in f if l.startswith("ATOM") or l.startswith("HETATM") or l.startswith("TER")]
            
    # Force chain ID
    final_lines = []
    for line in lines:
        new_line = list(line)
        if len(new_line) > 21:
            new_line[21] = chain_id
        final_lines.append("".join(new_line))
    return final_lines

def update_serial_numbers(lines, offset):
    """Updates atom serial numbers and CONECT records in PDB lines using fixed-width fields."""
    updated = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                serial = int(line[6:11])
                new_serial = serial + offset
                new_line = line[:6] + f"{new_serial:5d}" + line[11:]
                updated.append(new_line)
            except:
                updated.append(line)
        elif line.startswith("CONECT"):
            # CONECT records use fixed 5-column fields after the first 6 chars
            new_line = line[:6]
            for i in range(6, len(line.strip()), 5):
                field = line[i:i+5].strip()
                if field:
                    try:
                        new_val = int(field) + offset
                        new_line += f"{new_val:5d}"
                    except:
                        new_line += field.rjust(5)
                else:
                    new_line += "     "
            updated.append(new_line.rstrip().ljust(80) + "\n")
        else:
            updated.append(line)
    return updated

import gemmi

def add_coordination_conects(pdb_path):
    """
    Scans the PDB for metals and nearby O/N atoms, then inserts CONECT records before END.
    Filters out cross-structure bonds (Hit-to-Ref) to avoid visual clutter.
    """
    atoms = []
    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                serial = int(line[6:11])
                name = line[12:16].strip()
                chain = line[21]
                res_name = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip().upper()
                atoms.append({
                    'serial': serial,
                    'name': name,
                    'chain': chain,
                    'res_name': res_name,
                    'pos': (x, y, z),
                    'element': element,
                    'is_hit': chain.isupper()
                })
            except:
                continue

    metals = [a for a in atoms if a['element'] in ["ZN", "MG", "MN", "FE", "CA", "CU", "CO", "NI"]]
    others = [a for a in atoms if a['element'] in ["O", "N"]]
    
    conects = {}
    for m in metals:
        for o in others:
            # Only bond within the same structure (Hit-Hit or Ref-Ref)
            if m['is_hit'] != o['is_hit']:
                continue
            
            dist_sq = (m['pos'][0]-o['pos'][0])**2 + (m['pos'][1]-o['pos'][1])**2 + (m['pos'][2]-o['pos'][2])**2
            if dist_sq <= 2.8**2:
                if m['serial'] not in conects: conects[m['serial']] = []
                conects[m['serial']].append(o['serial'])

    new_lines = []
    # Insert CONECTs before END or at the end
    end_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("END"):
            end_idx = i
            break
    
    conect_lines = []
    for m_ser in sorted(conects.keys()):
        bonded = sorted(conects[m_ser])
        # PDB allows up to 4 partners per CONECT line
        for i in range(0, len(bonded), 4):
            chunk = bonded[i:i+4]
            line = f"CONECT{m_ser:5d}"
            for b_ser in chunk:
                line += f"{b_ser:5d}"
            conect_lines.append(line.ljust(80) + "\n")

    if end_idx != -1:
        new_lines = lines[:end_idx] + conect_lines + lines[end_idx:]
    else:
        new_lines = lines + conect_lines

    with open(pdb_path, 'w') as f:
        f.writelines(new_lines)

def extract_binding_sites(overlay_path, radius=10.0):
    """
    Identifies phosphate ligands in the overlay and extracts their 10A environment.
    Avoids redundant symmetrical sites by tracking unique (ligand_name, chain_sequence) pairs.
    """
    try:
        structure = gemmi.read_structure(overlay_path)
    except Exception as e:
        print(f"  [Error] Could not read overlay for extraction: {e}")
        return

    # 1. Map chain names to their protein sequences to identify identical chains (symmetrical subunits)
    chain_protein_seqs = {}
    standard_aa = set(["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", 
                       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"])
    
    for model in structure:
        for chain in model:
            if chain.name not in chain_protein_seqs:
                # Extract only standard amino acids for a robust sequence identity check
                aa_seq = [res.name for res in chain if res.name.upper() in standard_aa]
                chain_protein_seqs[chain.name] = "-".join(aa_seq)

    # 2. Identify unique phosphate binding sites
    # Key: (residue_name, protein_sequence_hash)
    seen_sites = set()
    extracted_count = 0

    base_dir = os.path.dirname(overlay_path)
    base_name = os.path.basename(overlay_path).replace(".pdb", "")

    for model in structure:
        for chain in model:
            chain_seq = chain_protein_seqs.get(chain.name, "")
            
            for res in chain:
                if res.name in ["HOH", "PO4", "PI", "2HP"]:
                    continue
                
                # Check for Phosphate
                has_p = any(atom.element.name == "P" for atom in res)
                if not has_p:
                    continue

                # Deduplication check
                site_key = (res.name, chain_seq)
                if site_key in seen_sites:
                    continue
                
                seen_sites.add(site_key)
                lig_positions = [atom.pos for atom in res]
                
                # Create a new structure for this environment
                new_st = gemmi.Structure()
                new_st.spacegroup_hm = structure.spacegroup_hm
                new_model = gemmi.Model(str(extracted_count + 1))
                
                # Extract 10A environment from ALL chains in the structure
                for m_idx, m in enumerate(structure):
                    for c in m:
                        selected_indices = set()
                        for r_idx, r in enumerate(c):
                            is_near = False
                            for atom in r:
                                for lp in lig_positions:
                                    if atom.pos.dist(lp) <= radius:
                                        is_near = True
                                        break
                                if is_near: break
                            if is_near:
                                selected_indices.add(r_idx)
                        
                        if not selected_indices:
                            continue

                        sorted_keep = sorted(list(selected_indices))
                        segments = []
                        if sorted_keep:
                            current_seg = [sorted_keep[0]]
                            for idx in sorted_keep[1:]:
                                if idx == current_seg[-1] + 1:
                                    current_seg.append(idx)
                                else:
                                    segments.append(current_seg)
                                    current_seg = [idx]
                            segments.append(current_seg)

                        for seg in segments:
                            new_chain = gemmi.Chain(c.name)
                            for r_idx in seg:
                                new_chain.add_residue(c[r_idx], -1)
                            new_model.add_chain(new_chain)
                
                if len(new_model) > 0:
                    new_st.add_model(new_model)
                    env_filename = f"env_{base_name}_{res.name}_{chain.name}_{res.seqid.num}.pdb"
                    env_path = os.path.join(base_dir, env_filename)
                    new_st.write_pdb(env_path)
                    add_coordination_conects(env_path)
                    print(f"    Saved unique environment: {env_filename}")
                    extracted_count += 1

    if extracted_count == 0:
        print("  [Info] No non-simple phosphate ligands found for environment extraction.")

def create_overlay(ref_id, hit_id, project_root, fast_mode=True):
    ref_id = ref_id.upper()
    hit_id = hit_id.upper()
    
    ref_dir = os.path.join(project_root, "data", "structures", "references")
    hits_dir = os.path.join(project_root, "data", "structures", "hits")
    output_dir = os.path.join(project_root, "results", "overlays")
    os.makedirs(output_dir, exist_ok=True)
    
    # Search in both hits and references for both structures
    ref_path = find_structure_file(ref_id, ref_dir) or find_structure_file(ref_id, hits_dir)
    hit_path = find_structure_file(hit_id, hits_dir) or find_structure_file(hit_id, ref_dir)
    
    if not ref_path or not hit_path:
        print(f"Error: Missing structure files for {ref_id} or {hit_id}")
        return False
        
    exec_dir = os.path.join(project_root, "scripts", "Shell scripts")
    if platform.system() == "Windows":
        tmalign_exec = os.path.join(exec_dir, "TMalign_linux")
    else:
        tmalign_exec = os.path.join(exec_dir, "TMalign")
        if not os.path.exists(tmalign_exec): tmalign_exec = os.path.join(exec_dir, "TMalign_linux")

    print(f"Calculating transformation matrix for {hit_id} onto {ref_id}...")
    res = run_tmalign(hit_path, ref_path, tmalign_executable=tmalign_exec, fast_mode=fast_mode)
    
    if not res or not res.get("transformation"):
        print("Error: Could not calculate transformation matrix.")
        return False
        
    matrix = res["transformation"]
    tm_score = res.get("tm_score_1", 0.0)
    
    # Assembly using Gemmi for robustness
    st_ref = gemmi.read_structure(ref_path)
    st_hit = gemmi.read_structure(hit_path)
    
    # Final structure
    final_st = gemmi.Structure()
    final_model = gemmi.Model("1")
    
    # 1. Process Hit - Apply transformation and Rename chains to 'A', 'B'...
    print(f"  Transforming hit structure {hit_id}...")
    # Use first model only
    h_model = st_hit[0]
    t = matrix['t']
    u = matrix['u']
    
    for i, h_chain in enumerate(h_model):
        new_chain = gemmi.Chain(chr(ord('A') + i)) # Rename to A, B, C...
        for res in h_chain:
            new_res = res.clone()
            for atom in new_res:
                x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
                nx = t[0] + u[0][0]*x + u[0][1]*y + u[0][2]*z
                ny = t[1] + u[1][0]*x + u[1][1]*y + u[1][2]*z
                nz = t[2] + u[2][0]*x + u[2][1]*y + u[2][2]*z
                atom.pos = gemmi.Position(nx, ny, nz)
                # Force standard 1-2 char element symbol (e.g. ZN2+ -> ZN)
                clean_el = "".join([c for c in atom.element.name if c.isalpha()])
                atom.element = gemmi.Element(clean_el[:2].upper())
            new_chain.add_residue(new_res, -1)
        final_model.add_chain(new_chain)
        
    # 2. Process Reference - Rename chains to 'a', 'b'... (lowercase)
    print(f"  Preparing reference structure {ref_id}...")
    r_model = st_ref[0]
    for i, r_chain in enumerate(r_model):
        new_chain = gemmi.Chain(chr(ord('a') + i)) # Rename to a, b, c...
        for res in r_chain:
            new_res = res.clone()
            for atom in new_res:
                # Force standard element symbol
                clean_el = "".join([c for c in atom.element.name if c.isalpha()])
                atom.element = gemmi.Element(clean_el[:2].upper())
            new_chain.add_residue(new_res, -1)
        final_model.add_chain(new_chain)
        
    final_st.add_model(final_model)
    final_output = os.path.join(output_dir, f"overlay_{ref_id}_{hit_id}.pdb")
    final_st.write_pdb(final_output)
    add_coordination_conects(final_output)
        
    print(f"Success! Overlay saved to: {final_output}")
    extract_binding_sites(final_output)
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate an overlay PDB of a hit structure onto a reference.")
    parser.add_argument("--ref", help="Reference PDB ID")
    parser.add_argument("--hit", help="Hit PDB ID")
    parser.add_argument("--list", action="store_true", help="List available pairs from results")
    parser.add_argument("--results", help="Path to TM-align results JSON")
    parser.add_argument("--slow", action="store_true", help="Run TM-align without -fast flag")
    
    args = parser.parse_args()
    
    results_file = args.results
    if not results_file:
        results_dir = os.path.join(PROJECT_ROOT, "results")
        json_files = glob.glob(os.path.join(results_dir, "tmalign_results_*.json"))
        if json_files: results_file = max(json_files, key=os.path.getmtime)
    
    if args.list:
        if not results_file or not os.path.exists(results_file):
            print("Error: Could not find results file to list pairs.")
            return
        with open(results_file, 'r') as f:
            data = json.load(f)
        print(f"Available reference-hit pairs (from {os.path.basename(results_file)}):")
        for ref, hits in list(data.items())[:20]:
            hit_ids = [h["hit"] for h in hits[:5]]
            print(f"  {ref}: {', '.join(hit_ids)}...")
        return

    if not args.ref or not args.hit:
        print("Usage: python create_overlay.py --ref <REF_ID> --hit <HIT_ID>")
        if results_file and os.path.exists(results_file):
             with open(results_file, 'r') as f:
                data = json.load(f)
                if data:
                    ref = list(data.keys())[0]
                    hit = data[ref][0]["hit"]
                    print(f"\nNo pair specified. Suggestion: --ref {ref} --hit {hit}")
        return

    create_overlay(args.ref, args.hit, PROJECT_ROOT, fast_mode=not args.slow)

if __name__ == "__main__":
    main()
