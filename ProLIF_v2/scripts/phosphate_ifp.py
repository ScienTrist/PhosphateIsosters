"""
ProLIF-cutoffs equivalent of analyze_plif.py's get_interactions_per_phosphate():
restricts a ligand's ProLIF fingerprint to just its phosphate group(s) -- P atom
plus O/N atoms within 2.1 A of it, same definition and same per-P-atom grouping
as the homebrew pipeline's get_interactions_per_phosphate() / phosphate_groups.
get_phosphate_groups() -- but detects the actual protein contacts using ProLIF's
own interaction classes (INTERACTIONS in run_prolif.py) and their geometric
cutoffs, not DISTANCE_CUTOFFS from src/constants.py.

Reuses run_prolif.py's pocket-loading and mol-building helpers (_extract_submol,
safe_molecule_from_mda, POCKETS_DIR, INTERACTIONS) rather than duplicating them;
importing run_prolif as a module is safe since its pipeline only runs under
`if __name__ == "__main__"`.

Ligand ionization (phosphate oxygens included) is now handled once, upstream,
in protonate_ligand.py via Dimorphite-DL at pH 7.4 -- lig_mol arrives here
already carrying the correct formal charges, so no per-group deprotonation
step is needed here anymore (this file used to have its own
_deprotonate_ionizable_oxygens for exactly that, phosphate-only; removed once
the upstream fix made it redundant -- see protonate_ligand.py's docstring).
"""
import os

import MDAnalysis as mda

import run_prolif as rp


def get_ligand_phosphate_groups(lig_mol):
    """RDKit-mol equivalent of phosphate_groups.get_phosphate_groups(): one group
    per P atom in lig_mol, each containing that P plus any O/N atom within 2.1 A
    of it. Returns [{"p_idx": i, "atom_idxs": [...]}, ...]; empty if no P atoms
    (e.g. the hit/mimic side, which by construction never has one)."""
    conf = lig_mol.GetConformer()
    p_atom_idxs = [a.GetIdx() for a in lig_mol.GetAtoms() if a.GetSymbol() == "P"]

    groups = []
    for p_idx, p_atom_idx in enumerate(p_atom_idxs):
        p_pos = conf.GetAtomPosition(p_atom_idx)
        atom_idxs = [p_atom_idx]
        for a in lig_mol.GetAtoms():
            if a.GetIdx() == p_atom_idx or a.GetSymbol() not in ("O", "N"):
                continue
            if p_pos.Distance(conf.GetAtomPosition(a.GetIdx())) < 2.1:
                atom_idxs.append(a.GetIdx())
        groups.append({"p_idx": p_idx, "atom_idxs": atom_idxs})
    return groups


def phosphate_group_ifps(site, fp):
    """Loads site's already-extracted pocket (POCKETS_DIR/{site_id}.pdb) and runs
    fp.generate() once per phosphate group of its ligand, submol-only vs the full
    protein. Returns ({p_idx: ifp_dict}, None) on success, or (None, err) if the
    pocket/ligand/protein selection fails. Empty dict (not an error) if the
    ligand has no phosphorus at all."""
    pocket_path = os.path.join(rp.POCKETS_DIR, f"{site['site_id']}.pdb")
    if not os.path.exists(pocket_path):
        return None, f"no extracted pocket at {pocket_path}"

    u = mda.Universe(pocket_path)
    # Not filtered by resname -- see run_prolif.py's run_site() for why (legacy PDB
    # format truncates 4-5 character extended CCD codes on every write in this
    # pipeline, so the manifest's full-length resname never matches these files).
    ligand = u.select_atoms(f"resid {site['lig_resnum']} and chainID {site['chain']} and not protein")
    protein = rp.protein_selection(u)  # metals + non-canonical/PTM backbone residues -- see run_prolif.py's protein_selection
    if len(ligand) == 0:
        return None, f"ligand selection matched 0 atoms in {pocket_path}"
    if len(protein) == 0:
        return None, f"protein selection matched 0 atoms in {pocket_path}"

    lig_mol = rp.safe_molecule_from_mda(ligand, use_segid=False)
    prot_mol = rp.safe_molecule_from_mda(protein, use_segid=False)

    groups = get_ligand_phosphate_groups(lig_mol)
    if not groups:
        return {}, None

    import prolif as plf
    from prolif.residue import Residue

    result = {}
    for group in groups:
        frag = rp._extract_submol(lig_mol, group["atom_idxs"])
        phospho_mol = plf.Molecule(frag, use_segid=False, residues=[Residue(frag, use_segid=False)])
        ifp = fp.generate(phospho_mol, prot_mol, residues=None, metadata=True)
        result[group["p_idx"]] = ifp
    return result, None


def count_interactions(ifp):
    """Total (protein_residue, interaction_type) hits in an ifp -- same convention
    run_prolif.py's run_site() uses to report n_bits for a whole-pocket ifp."""
    return sum(len(v) for v in ifp.values())


def best_phosphate_group(site, fp):
    """Runs phosphate_group_ifps() and returns (p_idx, ifp, n_interactions) for
    whichever phosphate group has the most detected interactions (ties broken by
    lowest p_idx), or (None, None, 0) if the ligand has no phosphorus or no group
    made any contact. This is the same "pick the busiest phosphate" convention
    compare_with_homebrew_plif.py uses for the homebrew side."""
    groups, err = phosphate_group_ifps(site, fp)
    if groups is None:
        return None, None, 0, err
    if not groups:
        return None, None, 0, None

    best_p_idx, best_ifp, best_n = None, None, -1
    for p_idx, ifp in groups.items():
        n = count_interactions(ifp)
        if n > best_n:
            best_p_idx, best_ifp, best_n = p_idx, ifp, n
    return best_p_idx, best_ifp, best_n, None
