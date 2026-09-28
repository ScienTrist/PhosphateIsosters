"""
Identifies individual phosphate groups within a (possibly multi-phosphate)
reference ligand, so PLIP's reference-side fingerprint can be restricted to
just the one group being tested against a given mimic -- matching how
analyze_plif.py's get_interactions_per_phosphate() scopes PLIF, instead of
scoring the mimic against the whole ligand's contacts (unfair for things like
IHP/inositol-hexakisphosphate with 6 groups, or NAD/FAD/ATP where the
phosphate is a small part of a much bigger cofactor).

A "group" = one P atom plus any O/N atoms covalently bonded to it (within
2.1A) -- same definition and same enumeration order (residue atom order) as
analyze_plif.py, so a given p_idx refers to the same physical group in both
pipelines.
"""
import os
import sys

import gemmi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from utils import dist as _dist  # noqa: E402


def get_phosphate_groups(structure, lig_name, lig_num):
    """Returns [{"p_idx": i, "atoms": [(x,y,z), ...]}, ...] for every P atom
    found in the ligand residue, in residue-atom-order (p_idx 0, 1, 2, ...)."""
    ligand_res = None
    for model in structure:
        for chain in model:
            for res in chain:
                if res.name == lig_name and res.seqid.num == lig_num:
                    ligand_res = res
                    break
            if ligand_res is not None:
                break
        break  # first model only

    if ligand_res is None:
        return []

    p_atoms = [a for a in ligand_res if a.element.name == "P"]
    groups = []
    for p_idx, p_atom in enumerate(p_atoms):
        group_atoms = [(p_atom.pos.x, p_atom.pos.y, p_atom.pos.z)]
        for a in ligand_res:
            if a.element.name in ("O", "N") and p_atom.pos.dist(a.pos) < 2.1:
                group_atoms.append((a.pos.x, a.pos.y, a.pos.z))
        groups.append({"p_idx": p_idx, "atoms": group_atoms})
    return groups


def filter_to_group(records, group_atoms, metal_cutoff=3.5, atom_cutoff=1.0):
    """Keeps only records whose ligand-side anchor (lig_coords) sits at/near
    one of the group's own atoms. METAL contacts use a looser cutoff since the
    metal ion itself isn't one of the group's atoms, just coordinated by them;
    every other type's lig_coords should be (near-)identical to one of the
    group atoms, since PLIP reports the actual interacting ligand atom."""
    out = []
    for r in records:
        if r.get("lig_coords") is None:
            continue
        cutoff = metal_cutoff if r["type"] == "METAL" else atom_cutoff
        if any(_dist(r["lig_coords"], ga) <= cutoff for ga in group_atoms):
            out.append(r)
    return out
