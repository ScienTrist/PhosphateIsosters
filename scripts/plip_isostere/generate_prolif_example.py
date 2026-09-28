"""
Generates a ProLIF interaction fingerprint for one ligand binding site --
an intermediate-step artifact to compare directly against PLIP's native
report for the identical site, using the identical protonated geometry (both
tools run on the same polar-hydrogens-added structure PLIP itself produces,
so differences reflect each tool's interaction *model*, not differing input
hydrogens).

Unlike PLIP, ProLIF has no built-in protonation step -- it expects explicit
hydrogens to already be present for its default HBDonor/HBAcceptor SMARTS.
Reusing PLIP's own `{basename}_protonated.pdb` (written by
plip.structure.preparation.PDBComplex.load_pdb via OpenBabel's
AddPolarHydrogens()) keeps both tools' fingerprints comparable instead of
introducing a second, differently-protonated structure.

Run directly to regenerate the 10GW (reference) / 4E21 (hit) example pair;
call generate_report() with other (pdb_id, category, ligand) args for a
different site.
"""
import json
import os
import sys
from collections import defaultdict
from operator import attrgetter

import MDAnalysis as mda
import numpy as np
from rdkit import Chem

from plip.structure.preparation import PDBComplex, create_folder_if_not_exists

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT_DIR, find_pdb_path

REPORT_DIR = os.path.join(OUT_DIR, "example_report")

# Roughly PLIP's interaction coverage: Hydrophobic, H-bonds, salt bridges
# (Anionic/Cationic), pi-stacking, pi-cation, halogen bonds (XB), metal
# coordination. ProLIF's HBDonor/HBAcceptor require explicit hydrogens on the
# donor (which we now have via the protonated PDB), so we use those rather
# than the no-hydrogen ImplicitHBDonor/ImplicitHBAcceptor variants.
INTERACTIONS = [
    "Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking",
    "Anionic", "Cationic", "CationPi", "PiCation",
    "XBDonor", "XBAcceptor", "MetalDonor", "MetalAcceptor", "VdWContact",
]


def _json_safe(value):
    """Recursively converts numpy scalars/tuples inside raw ProLIF metadata
    (indices, parent_indices, distance, angle, ...) into plain JSON types."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atom_info(atomgroup, idx):
    """Resolves a parent_indices position (an index into the ligand/protein
    AtomGroup passed to fp.run) into a human-readable atom identity -- raw
    RDKit atom indices alone aren't useful for debugging without this."""
    a = atomgroup.atoms[idx]
    return {
        "name": a.name, "element": getattr(a, "element", ""),
        "resname": a.resname, "resid": int(a.resid), "chain": a.segid,
        "serial": int(a.id) if hasattr(a, "id") else None,
        "pos": [float(c) for c in a.position],
    }


def safe_molecule_from_mda(atomgroup, use_segid=False, **converter_kwargs):
    """Drop-in replacement for prolif.Molecule.from_mda() that avoids a
    native segfault observed in RDKit 2025.09.5's GetMolFrags(asMols=True)
    (called by prolif.utils.split_mol_by_residues -> SplitMolByPDBResidues
    + GetMolFrags) on some structures -- reproduced via manual bisection on
    4E21's binding-site selection, where GetMolFrags crashes splitting the
    per-resname-grouped GLU fragment into per-instance residues, even though
    converting the same selection to RDKit and converting each residue
    individually both succeed.

    Builds each residue as its own submol by copying out just its atoms/bonds
    (via _extract_submol below) from manually grouped (resname, resid, chain,
    icode) atom indices, then feeds them to Molecule's `residues=` bypass
    (prolif >= 2.2.0) so split_mol_by_residues/GetMolFrags is never invoked.
    mapindex must be stamped before splitting (mirroring what
    Molecule.__init__ does) so parent_indices still resolve correctly back
    onto `atomgroup`.
    """
    mol = atomgroup.convert_to.rdkit(**converter_kwargs)
    for atom in mol.GetAtoms():
        atom.SetUnsignedProp("mapindex", atom.GetIdx())

    residue_atom_idxs = defaultdict(list)
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        key = (info.GetResidueName(), info.GetResidueNumber(), info.GetChainId(), info.GetInsertionCode())
        residue_atom_idxs[key].append(atom.GetIdx())

    import prolif as plf
    from prolif.residue import Residue
    residues = [Residue(_extract_submol(mol, idxs), use_segid=use_segid)
                for idxs in residue_atom_idxs.values()]
    residues.sort(key=attrgetter("resid"))
    return plf.Molecule(mol, use_segid=use_segid, residues=residues)


def _extract_submol(mol, idxs):
    """Atom-induced subgraph of `mol` on `idxs`, preserving atom properties
    (incl. mapindex, PDB monomer info) and 3D coordinates. NOT the same as
    Chem.PathToSubmol, which takes a path of BOND indices, not atom indices
    -- using it with atom indices silently returns the wrong atoms entirely
    (confirmed: it returned a neighboring GLU residue's atoms when asked for
    GLY280's atom indices). Chem.RWMot(mol) + RemoveAtom per residue also
    works but is O(n_atoms) per call since it copies the whole parent mol
    each time -- for a ~20000-atom protein split into ~1300 residues that's
    ~10s; this AddAtom/AddBond version is O(residue size) per call instead.
    """
    conf = mol.GetConformer()
    frag = Chem.RWMol()
    old_to_new = {}
    for old_idx in idxs:
        old_to_new[old_idx] = frag.AddAtom(mol.GetAtomWithIdx(old_idx))
    idx_set = set(idxs)
    seen_bonds = set()
    for old_idx in idxs:
        for bond in mol.GetAtomWithIdx(old_idx).GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in idx_set and j in idx_set:
                key = (min(i, j), max(i, j))
                if key not in seen_bonds:
                    seen_bonds.add(key)
                    frag.AddBond(old_to_new[i], old_to_new[j], bond.GetBondType())
    new_conf = Chem.Conformer(len(idxs))
    for old_idx, new_idx in old_to_new.items():
        new_conf.SetAtomPosition(new_idx, conf.GetAtomPosition(old_idx))
    frag_mol = frag.GetMol()
    frag_mol.AddConformer(new_conf, assignId=True)
    # AddAtom() copies each atom's symbol/properties/monomer-info, but not
    # RDKit's internally cached explicit-valence data computed by the parent
    # mol's own sanitization -- without re-sanitizing here, later SMARTS
    # matching (GetSubstructMatches, used by every interaction type) raises
    # "getValence(EXPLICIT) called without call to calcExplicitValence()".
    try:
        Chem.SanitizeMol(frag_mol)
    except Chem.KekulizeException:
        # Proline residues (no backbone N-H under PLIP's polar-only
        # protonation) had their saturated, non-aromatic ring mis-flagged as
        # aromatic by the parent mol's whole-structure bond-order inferrer;
        # kekulizing that ring in isolation then fails. The ring isn't
        # actually aromatic, so skipping kekulization/aromaticity here (but
        # keeping ring/valence perception, needed by SMARTS matching) is the
        # chemically correct fix, not just an error swallow.
        ops = Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY
        Chem.SanitizeMol(frag_mol, sanitizeOps=ops)
    return frag_mol


def ensure_protonated_pdb(pdb_id, pdb_path):
    """Reuses PLIP's own protonation step so both tools see the same
    hydrogens. Returns the path to `{pdb_id}_protonated.pdb`."""
    protonated_path = os.path.join(REPORT_DIR, f"{pdb_id}_protonated.pdb")
    if os.path.exists(protonated_path):
        return protonated_path
    create_folder_if_not_exists(REPORT_DIR)
    mol = PDBComplex()
    mol.output_path = REPORT_DIR + os.sep
    mol.load_pdb(pdb_path)  # writes {basename}_protonated.pdb as a side effect
    return protonated_path


def generate_report(pdb_id, category, lig_resname, lig_chain, lig_resnum):
    """category is "references" or "hits" (see common.find_pdb_path)."""
    pdb_path = find_pdb_path(pdb_id, category)
    if pdb_path is None:
        raise SystemExit(f"{pdb_id}: no .pdb file found under data/structures/{category}")

    protonated_path = ensure_protonated_pdb(pdb_id, pdb_path)

    u = mda.Universe(protonated_path)
    ligand = u.select_atoms(f"resname {lig_resname} and segid {lig_chain} and resid {lig_resnum}")
    # ProLIF's Residue class stores resid as np.uint32, so structures with
    # negative residue numbers crash on conversion (e.g. an N-terminal tag
    # numbered -2/-1 ahead of residue 1) -- dropped rather than renumbering.
    protein = u.select_atoms("protein and resid 1:99999")
    if len(ligand) == 0:
        raise SystemExit(f"ligand selection matched 0 atoms in {protonated_path}")

    import prolif as plf
    # use_segid=False: without it, ProLIF auto-detects more MDAnalysis
    # segments than PDB chains (the protonated file has chain breaks) and
    # falls back to labeling residues by segment index instead of chain
    # letter (e.g. "12" instead of "A") -- forcing real chain IDs keeps
    # residue labels comparable to PLIP's report.
    #
    # fp.run() is avoided here -- it calls Molecule.from_mda() internally,
    # which segfaults on some structures (see safe_molecule_from_mda's
    # docstring). Building the Molecules ourselves and calling fp.generate()
    # directly is the same computation, just without the crashing step.
    fp = plf.Fingerprint(INTERACTIONS, use_segid=False)
    lig_mol = safe_molecule_from_mda(ligand, use_segid=False)
    prot_mol = safe_molecule_from_mda(protein, use_segid=False)
    ifp = fp.generate(lig_mol, prot_mol, residues=None, metadata=True)
    fp.ifp = {0: ifp}

    df = fp.to_dataframe()
    csv_path = os.path.join(REPORT_DIR, f"{pdb_id}_prolif.csv")
    df.to_csv(csv_path)

    # Full dump: every raw metadata field ProLIF computed per contact
    # (distance, angle, offset, ... -- whatever the interaction class returns,
    # not just "distance"), plus parent_indices resolved to actual atom
    # identities (including xyz position) via the same ligand/protein
    # AtomGroups passed to fp.run. `to_dataframe()` only keeps a
    # boolean/count bit per contact and drops all of this.
    records = []
    for data in fp.ifp[0].interactions():
        meta = _json_safe(data.metadata)
        record = {
            "protein_residue": str(data.protein),
            "ligand_residue": str(data.ligand),
            "interaction": data.interaction,
            "metadata": meta,
        }
        indices = data.metadata.get("parent_indices", {})
        if "ligand" in indices:
            record["ligand_atoms"] = [_atom_info(ligand, i) for i in indices["ligand"]]
        if "protein" in indices:
            record["protein_atoms"] = [_atom_info(protein, i) for i in indices["protein"]]
        records.append(record)
    records.sort(key=lambda r: (r["protein_residue"], r["interaction"]))

    json_path = os.path.join(REPORT_DIR, f"{pdb_id}_prolif_full.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    # Flat per-contact text report (all metadata fields, resolved atom
    # names), formatted to sit next to PLIP's own {pdb_id}_report.txt for a
    # direct side-by-side comparison.
    txt_path = os.path.join(REPORT_DIR, f"{pdb_id}_prolif_report.txt")
    with open(txt_path, "w") as f:
        f.write(f"ProLIF interaction fingerprint for PDB structure {pdb_id}\n")
        f.write("=" * 60 + "\n")
        f.write(f"ProLIF v{plf.__version__}\n")
        f.write(f"Ligand: {lig_resname}:{lig_chain}:{lig_resnum}\n")
        f.write(f"Interaction types: {', '.join(INTERACTIONS)}\n\n")

        for r in records:
            f.write(f"{r['protein_residue']}  <->  {r['ligand_residue']}  [{r['interaction']}]\n")
            for key, val in r["metadata"].items():
                if key in ("indices", "parent_indices"):
                    continue
                if isinstance(val, float):
                    f.write(f"    {key}: {val:.3f}\n")
                else:
                    f.write(f"    {key}: {val}\n")
            for side in ("protein_atoms", "ligand_atoms"):
                if side not in r:
                    continue
                names = ", ".join(f"{a['name']}({a['element']})" for a in r[side])
                f.write(f"    {side}: {names}\n")
            f.write("\n")
        f.write(f"Total interactions: {len(records)}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")
    print(f"Wrote {json_path}")
    print(f"{pdb_id}: {len(records)} interactions\n")


if __name__ == "__main__":
    # 10GW/6PG:A:401 (reference) paired with 4E21/MRD:A:401 (hit) -- the
    # same pair resolve_pairs.resolve_row() resolves for this isostere
    # candidate (see results/motif_analysis/isostere_full_list.csv).
    generate_report("10GW", "references", "6PG", "A", 401)
    generate_report("4E21", "hits", "MRD", "A", 401)
