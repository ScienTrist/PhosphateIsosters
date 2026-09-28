"""
extract_multiphosphate_binding_motifs.py -- companion to extract_binding_motifs.py,
for the subset of reference ligands that carry MORE THAN ONE phosphate group
(ADP, ATP, NAD, CoA, GTP analogues, ... up to IHP with 6 and I7P with 7).

Why a separate script instead of extending extract_binding_motifs.py: for a
mono-phosphate ligand, "residues touching phosphate group P1" and "residues
touching the whole ligand" are the same set -- extract_binding_motifs.py
already covers that case, and re-running it per phosphate group would just
duplicate its output. This script only has something new to say for the
multi-phosphate subset (confirmed 388/658 reference sites, 102 ligand codes,
via a one-off P-atom count over pocket_types/results/aa_type_overlay_assignments.csv
against ProLIF_v2/data/pockets/*.pdb).

Phosphate-group decomposition: ligand atoms are grouped around each
phosphorus atom by 3D BONDING DISTANCE (<=1.85 A), not by PDB atom-name
convention (PA/PB/PG) -- the naming convention holds for standard
nucleotides but isn't guaranteed for arbitrary multi-phosphate ligands
(e.g. IHP's inositol ring phosphates). A bridging oxygen (bonded to two
phosphorus atoms at once, e.g. ADP's PA-O-PB linkage) ends up a member of
BOTH neighbouring groups by construction -- this is intentional, not a bug.

Residue-to-group attribution uses the SAME cached ProLIF fingerprint as
build_pocket_features.py / extract_binding_motifs.py, at ATOM granularity
(interaction metadata's parent_indices.ligand, confirmed to match the
pocket PDB file's own HETATM atom order for the reference ligand instance).
A residue contacting atoms from two different phosphate groups (a metal
bridging beta/gamma, or a long side chain reaching across alpha/beta) is
listed under BOTH groups' motif windows -- confirmed with the user this is
preferred over an arbitrary nearest-group tie-break, since collapsing a
genuine bridge to one side would misrepresent the actual structure.

Sequence-window extraction (padding/gap-filling, uppercase/lowercase
convention, '-' for unresolved) is identical to extract_binding_motifs.py
and imported from it rather than reimplemented.

Output is grouped by LIGAND CODE, not by pocket-type cluster: the question
this script answers is "how is each phosphate position of ligand X
recognised across all structures that bind it", independent of which
geometric pocket-type cluster a given site's whole pocket happens to fall
into.

Known limitation: 5 reference sites use post-2023 5-character PDB chemical
component IDs (e.g. A1INE) that don't fit the legacy fixed-column HETATM
resname field this script (and the wider pocket_types pipeline) parses --
they're skipped with a note, not silently miscounted.

Usage: python extract_multiphosphate_binding_motifs.py [--assignments path]
                                                          [--fp-pickle path]
                                                          [--pockets-dir path]
                                                          [--structures-dir path]
                                                          [--out path]
"""
import argparse
import csv
import math
import os
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POCKET_TYPES_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(POCKET_TYPES_ROOT)
PROLIF_V2_ROOT = os.path.join(PROJECT_ROOT, "ProLIF_v2")
RESULTS_DIR = os.path.join(POCKET_TYPES_ROOT, "results")

sys.path.insert(0, SCRIPT_DIR)
from build_pocket_features import _load_ifp_by_site, DEFAULT_EXCLUDED_INTERACTIONS  # noqa: E402
from extract_binding_motifs import (  # noqa: E402
    find_structure_path, load_chain_sequence, group_and_pad, render_window,
    STRUCTURES_DIR_DEFAULT,
)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from constants import METALS  # noqa: E402

POCKETS_DIR_DEFAULT = os.path.join(PROLIF_V2_ROOT, "data", "pockets")
BOND_DISTANCE = 1.85  # Angstrom cutoff for "these two ligand atoms are bonded"


def parse_ligand_atoms(pocket_pdb_path, ligand_code):
    """Returns [(atom_name, element, (x, y, z)), ...] in file order (== the
    ProLIF interaction metadata's ligand atom index order, confirmed against
    a real fingerprint entry) for the FIRST (chain, resSeq) instance of
    ligand_code encountered -- pocket files are ~10A single-site extractions,
    so a second distinct copy of the same ligand type is not expected, but
    this guards against it rather than silently merging two instances."""
    atoms = []
    target_key = None
    with open(pocket_pdb_path) as f:
        for line in f:
            if not line.startswith(("HETATM", "ATOM  ")):
                continue
            resname = line[17:20].strip()
            if resname != ligand_code:
                continue
            key = (line[21], line[22:26].strip())
            if target_key is None:
                target_key = key
            elif key != target_key:
                continue
            name = line[12:16].strip()
            element = line[76:78].strip() or (name[0] if name else "")
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            atoms.append((name, element, xyz))
    return atoms


def group_phosphates(atoms):
    """Returns [(label, atom_index_set), ...], one entry per phosphorus atom
    found in atoms, ordered by first appearance. label is the P atom's own
    PDB atom name (e.g. "PB") for readability. Group membership is by 3D
    bonding distance, so a bridging oxygen belongs to two groups at once --
    see module docstring."""
    def dist(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    p_indices = [i for i, (name, element, _) in enumerate(atoms) if element == "P"]
    groups = []
    for p_idx in p_indices:
        p_name, _, p_xyz = atoms[p_idx]
        members = {p_idx}
        for j, (name, element, xyz) in enumerate(atoms):
            if j == p_idx:
                continue
            if dist(p_xyz, xyz) <= BOND_DISTANCE:
                members.add(j)
        groups.append((p_name, members))
    return groups


def residues_by_phosphate_group(ifp, groups, excluded_interactions):
    """Returns {group_label: {chain: set(resnum)}, ...}. A residue lands in
    every group whose atom-index set it has a qualifying (non-excluded)
    interaction against -- deliberately not exclusive, see module docstring
    on bridging contacts. Metals excluded (not part of the polypeptide
    sequence), matching extract_binding_motifs.py's convention."""
    by_group = {label: defaultdict(set) for label, _ in groups}
    for (lig_id, prot_id), interactions in ifp.items():
        if prot_id.name in METALS:
            continue
        for name, mds in interactions.items():
            if name in excluded_interactions:
                continue
            for md in mds:
                lig_atom_idxs = md.get("parent_indices", {}).get("ligand", ())
                for label, members in groups:
                    if members & set(lig_atom_idxs):
                        by_group[label][prot_id.chain].add(prot_id.number)
    return by_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", default=os.path.join(RESULTS_DIR, "aa_type_overlay_assignments.csv"))
    parser.add_argument("--fp-pickle", default=os.path.join(PROLIF_V2_ROOT, "results",
                                                              "prolif_fingerprint_full_manifest.pkl"))
    parser.add_argument("--pockets-dir", default=POCKETS_DIR_DEFAULT)
    parser.add_argument("--structures-dir", default=STRUCTURES_DIR_DEFAULT)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "multiphosphate_binding_motifs.txt"))
    args = parser.parse_args()

    with open(args.assignments, newline="") as f:
        rows = list(csv.DictReader(f))

    by_ligand = defaultdict(list)
    for r in rows:
        by_ligand[r["ref_ligand"]].append(r)

    print(f"Loading cached ProLIF fingerprint from {args.fp_pickle} ...")
    ifp_by_site = _load_ifp_by_site(args.fp_pickle)

    seq_cache = {}
    n_sites, n_skipped_mono, n_skipped_other = 0, 0, 0
    t0 = time.time()

    with open(args.out, "w") as out_f:
        for lig in sorted(by_ligand, key=lambda l: -len(by_ligand[l])):
            entries = by_ligand[lig]

            # decide multi-phosphate status from the FIRST resolvable instance
            n_p = None
            for r in entries:
                sid = r["site_id"]
                base = sid[4:] if sid.startswith("ref_") else sid
                pocket_path = os.path.join(args.pockets_dir, f"ref_{base}.pdb")
                if not os.path.exists(pocket_path):
                    pocket_path = os.path.join(args.pockets_dir, f"hit_{base}.pdb")
                if not os.path.exists(pocket_path):
                    continue
                atoms = parse_ligand_atoms(pocket_path, lig)
                n_p = sum(1 for _, element, _ in atoms if element == "P")
                if n_p:
                    break
            if n_p is None:
                n_skipped_other += len(entries)
                continue
            if n_p < 2:
                n_skipped_mono += len(entries)
                continue

            out_f.write("=" * 80 + "\n")
            out_f.write(f"LIGAND: {lig}  ({len(entries)} sites, {n_p} phosphate groups)\n")
            out_f.write("=" * 80 + "\n\n")

            for r in entries:
                sid, pdb_id = r["site_id"], r.get("pdb_id", "")
                ifp = ifp_by_site.get(sid)
                if not ifp:
                    out_f.write(f">{sid}  [{pdb_id}] -- no cached fingerprint, skipped\n\n")
                    continue

                base = sid[4:] if sid.startswith("ref_") else sid
                pocket_path = os.path.join(args.pockets_dir, f"ref_{base}.pdb")
                if not os.path.exists(pocket_path):
                    pocket_path = os.path.join(args.pockets_dir, f"hit_{base}.pdb")
                if not os.path.exists(pocket_path):
                    out_f.write(f">{sid}  [{pdb_id}] -- pocket file not found, skipped\n\n")
                    continue

                atoms = parse_ligand_atoms(pocket_path, lig)
                groups = group_phosphates(atoms)
                if len(groups) < 2:
                    out_f.write(f">{sid}  [{pdb_id}] -- only {len(groups)} resolved P atom(s) in this "
                                f"instance, skipped\n\n")
                    continue

                by_group = residues_by_phosphate_group(ifp, groups, DEFAULT_EXCLUDED_INTERACTIONS)

                struct_path = find_structure_path(pdb_id, args.structures_dir)
                if struct_path is None:
                    out_f.write(f">{sid}  [{pdb_id}] -- structure file not found, skipped\n\n")
                    continue

                out_f.write(f">{sid}  [{pdb_id}]\n")
                for label, _ in groups:
                    by_chain = by_group[label]
                    if not by_chain:
                        out_f.write(f"  [{label}]  no non-metal interacting residues\n")
                        continue
                    for chain_id, positions in sorted(by_chain.items()):
                        cache_key = (pdb_id, chain_id)
                        if cache_key not in seq_cache:
                            seq_cache[cache_key] = load_chain_sequence(struct_path, chain_id)
                        seq = seq_cache[cache_key]
                        if not seq:
                            out_f.write(f"  [{label}] chain {chain_id}: no sequence extracted\n")
                            continue
                        chain_lo, chain_hi = min(seq), max(seq)
                        for lo, hi, binding in group_and_pad(positions):
                            lo, hi = max(lo, chain_lo), min(hi, chain_hi)
                            window = render_window(seq, lo, hi, binding)
                            out_f.write(f"  [{label}] chain {chain_id} {lo}-{hi}: {window}\n")
                out_f.write("\n")

                n_sites += 1
                if n_sites % 100 == 0:
                    print(f"  [{n_sites}] sites written ({time.time()-t0:.0f}s elapsed)")

    print(f"\nDone: {n_sites} multi-phosphate sites written, "
          f"{n_skipped_mono} mono-phosphate sites skipped (see extract_binding_motifs.py output instead), "
          f"{n_skipped_other} sites skipped (no pocket file / unresolved ligand), "
          f"in {time.time()-t0:.0f}s")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
