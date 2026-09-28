import os
import glob
import json


def load_ec_map(motif_dir):
    """Loads PDB-to-EC-class mapping from the ligand_search_*.json file."""
    parent_dir = os.path.dirname(motif_dir)
    ls_files = glob.glob(os.path.join(parent_dir, "ligand_search_*.json"))
    ec_map = {}
    if ls_files:
        ls_file = ls_files[0]
        try:
            with open(ls_file, "r") as f:
                ls_data = json.load(f)
                for major_ec, subs in ls_data.get("grouped_by_uniprot", {}).items():
                    for sub_ec, up_ids in subs.items():
                        for up_id, up_data in up_ids.items():
                            for pdb in up_data.get("pdbs", []):
                                ec_map[pdb.upper()] = major_ec
        except Exception as e:
            print(f"Warning: Could not load EC map from {ls_file}: {e}")
    return ec_map


def is_metal_coordinated(st_ref, ref_lig_name, ref_lig_num):
    """Checks if the reference phosphate is within 2.8 Å of a metal ion."""
    metals = {"ZN", "MG", "MN", "FE", "CA", "CU", "CO", "NI"}
    lig_atoms = []
    for model in st_ref:
        for chain in model:
            for res in chain:
                if res.name == ref_lig_name and res.seqid.num == int(ref_lig_num):
                    lig_atoms = [a.pos for a in res]
                    break

    if not lig_atoms:
        return False

    for model in st_ref:
        for chain in model:
            for res in chain:
                if res.name.upper() in metals:
                    for m_atom in res:
                        if any(m_atom.pos.dist(l_pos) <= 2.8 for l_pos in lig_atoms):
                            return True
    return False
