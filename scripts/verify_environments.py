import gemmi
import os
import numpy as np
import glob

def calculate_local_rmsd(env_path):
    """
    Calculates the RMSD of matched CA atoms between the Hit (uppercase)
    and Reference (lowercase) within the environment file.
    """
    try:
        st = gemmi.read_structure(env_path)
    except:
        return None, "Read Error"
    
    m = st[0]
    # In our files: Hit is A, B, C... | Reference is a, b, c...
    hit_atoms = {}
    ref_atoms = {}
    
    for chain in m:
        is_hit = chain.name.isupper()
        for res in chain:
            if not gemmi.find_tabulated_residue(res.name).is_amino_acid():
                continue
            ca = res.find_atom("CA", '\0')
            if ca:
                # Key by residue name and seqid to match homologous residues
                # Note: This assumes homologous residue numbering is similar or we rely on the alignment
                # For high TM-scores, we match by sequence position if possible
                key = (res.name, res.seqid.num)
                if is_hit:
                    hit_atoms[key] = ca.pos
                else:
                    ref_atoms[key] = ca.pos
    
    # Match pairs
    common_keys = set(hit_atoms.keys()).intersection(set(ref_atoms.keys()))
    if len(common_keys) < 3:
        return 0.0, f"Too few matching residues ({len(common_keys)})"
    
    hit_pts = np.array([[hit_atoms[k].x, hit_atoms[k].y, hit_atoms[k].z] for k in common_keys])
    ref_pts = np.array([[ref_atoms[k].x, ref_atoms[k].y, ref_atoms[k].z] for k in common_keys])
    
    diff = hit_pts - ref_pts
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd, len(common_keys)

def verify_all_environments(overlay_dir):
    env_files = glob.glob(os.path.join(overlay_dir, "env_*.pdb"))
    print(f"{'Environment File':<60} | {'RMSD':<8} | {'Matched CA'}")
    print("-" * 85)
    
    for env in env_files:
        rmsd, info = calculate_local_rmsd(env)
        name = os.path.basename(env)
        if isinstance(info, str):
            print(f"{name:<60} | {'N/A':<8} | {info}")
        else:
            print(f"{name:<60} | {rmsd:8.3f} | {info}")

if __name__ == "__main__":
    PROJECT_ROOT = "D:\\AI\\Master_thesis\\Phosphate-binding-site"
    verify_all_environments(os.path.join(PROJECT_ROOT, "results", "overlays"))
