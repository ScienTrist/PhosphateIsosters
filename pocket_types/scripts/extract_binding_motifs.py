"""
extract_binding_motifs.py -- Step X of pocket_types: for every AA-type pocket
type, and every reference site in it, extracts the LOCAL SEQUENCE CONTEXT
(1-letter code, padded a few residues each way) around that site's own
binding residues, for manual motif recognition (e.g. spotting a P-loop-style
consensus across a pocket type's members) -- something none of this
project's existing outputs provide, since everywhere else in pocket_types
works in 3D coordinates, never in raw sequence.

Binding residues: the SAME definition used everywhere else in this project
(site_typed_points, pocket_selection_by_type) -- every amino acid residue
with at least one qualifying (non-VdWContact) ProLIF interaction with the
reference ligand. Metals are excluded here specifically because they aren't
part of the polypeptide chain/sequence at all, unlike everywhere else in
pocket_types where they're a first-class citizen of the comparison.

Padding/gap-filling rule (confirmed against worked examples, not guessed;
padding widened from the original 2/3 to 3/5 to make more room for
literature-style motifs whose flanking residues -- e.g. the first Gly of a
Walker A GxxxxGK[S/T] -- often aren't themselves flagged as interacting):
  - Sort a chain's binding-residue positions. Two consecutive positions are
    CONNECTED if at most 3 residues are missing between them (i.e.
    B - A - 1 <= 3); connections chain transitively into groups.
  - An ISOLATED (size-1) group is padded by exactly 3 residues each way.
  - A multi-residue group has every internal gap filled completely (already
    guaranteed <=3 missing by construction), then each OUTER edge is
    extended by min(5, raw distance to its own inward neighbor) -- NOT a
    flat +5, and NOT relative to the group's overall span. E.g. {1,3,5} ->
    1-7 (residue 5's inward neighbor is 3, distance 2, so +2 not +5);
    {1,5,9} -> 1-13 (residue 9's inward neighbor is 5, distance 4, so
    min(5,4)=+4). Both ends are then clipped to the chain's own actually-
    resolved residue range -- padding never invents residues that were
    disordered/unresolved in the crystal structure, or don't exist at all.

Output convention (plain text, grouped by pocket type then by site):
  - Binding residues are UPPERCASE, padding/context residues are lowercase.
  - A position with no resolved coordinate in the structure (a disordered
    loop, not a numbering artifact) renders as '-', so padding never
    silently pretends a real gap in the crystal structure isn't there.

Sequences come from the FULL, uncropped reference structure files (data/
structures/references/*.pdb, mirrored from ProLIF_v2/data/raw) -- NOT the
~10A pocket-extracted files ProLIF_v2/data/pockets uses elsewhere in this
project, since a sequence neighbor +/-3 positions away isn't guaranteed to
be spatially within a 10A radius of the ligand. Confirmed complete (0
missing) for every reference PDB ID in the current manifest.

Known simplifications, both confirmed rare in this dataset (see project
history): alternate conformations are deduplicated by keeping whichever
copy gemmi's residue iteration yields first; a residue with a PDB insertion
code collapses onto its own base integer position (last one seen wins) --
acceptable here since this output is for visual motif recognition, not a
numerically exact positional index.

Usage: python extract_binding_motifs.py [--assignments path] [--labels-csv path]
                                          [--fp-pickle path] [--structures-dir path]
                                          [--out path]
"""
import argparse
import os
import sys
import time
import warnings
from collections import defaultdict

import gemmi

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POCKET_TYPES_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(POCKET_TYPES_ROOT)
PROLIF_V2_ROOT = os.path.join(PROJECT_ROOT, "ProLIF_v2")
RESULTS_DIR = os.path.join(POCKET_TYPES_ROOT, "results")

sys.path.insert(0, SCRIPT_DIR)
from build_pocket_features import _load_ifp_by_site, DEFAULT_EXCLUDED_INTERACTIONS  # noqa: E402
from build_cluster_overlays import load_cluster_members, load_cluster_labels  # noqa: E402

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from constants import METALS  # noqa: E402

STRUCTURES_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "data", "structures", "references")

CONNECT_GAP_MAX = 3   # missing-residue count; two binding residues connect if <=3 apart
ISOLATED_PAD = 3      # each-way padding for a binding residue with no connected neighbor
EDGE_PAD_MAX = 5       # cap on each-way padding extending PAST a connected group's outer edge


def find_structure_path(pdb_id, structures_dir):
    for ext in (".pdb", ".cif"):
        p = os.path.join(structures_dir, f"{pdb_id.upper()}{ext}")
        if os.path.exists(p):
            return p
    return None


def load_chain_sequence(struct_path, chain_id):
    """Returns {resnum: one_letter_code} for one chain of one structure.
    Skips non-amino-acid residues (ligands, waters, metals -- anything
    gemmi's own chemical-component table doesn't classify as an amino
    acid). Duplicate resnums (alternate conformations, or an insertion-
    coded residue collapsing onto its base integer) keep whichever the
    iteration reaches first -- see module docstring for why that's an
    acceptable simplification here."""
    st = gemmi.read_structure(struct_path)
    seq = {}
    for model in st:
        for chain in model:
            if chain.name != chain_id:
                continue
            for res in chain:
                info = gemmi.find_tabulated_residue(res.name)
                if info is None or not info.is_amino_acid():
                    continue
                resnum = res.seqid.num
                if resnum in seq:
                    continue
                one_letter = info.one_letter_code.upper()
                seq[resnum] = one_letter if one_letter.isalpha() else "X"
        break  # first model only
    return seq


def binding_residues_by_chain(ifp, excluded_interactions):
    """Returns {chain: set(resnum), ...} for every AMINO ACID residue (metals
    excluded -- see module docstring) that made at least one qualifying
    interaction with the reference ligand -- the same "interacting residue"
    definition used everywhere else in pocket_types."""
    by_chain = defaultdict(set)
    for (lig_id, prot_id), interactions in ifp.items():
        if prot_id.name in METALS:
            continue
        qualifying = [name for name, mds in interactions.items() if name not in excluded_interactions and mds]
        if qualifying:
            by_chain[prot_id.chain].add(prot_id.number)
    return by_chain


def group_and_pad(positions):
    """Implements the module docstring's confirmed gap-filling/padding rule.
    Returns [(lo, hi, binding_set), ...], one tuple per group, UNCLIPPED --
    callers must clip (lo, hi) to the chain's own actually-resolved residue
    range before rendering."""
    positions = sorted(positions)
    groups = [[positions[0]]]
    for p in positions[1:]:
        if p - groups[-1][-1] - 1 <= CONNECT_GAP_MAX:
            groups[-1].append(p)
        else:
            groups.append([p])

    result = []
    for group in groups:
        binding = set(group)
        if len(group) == 1:
            lo, hi = group[0] - ISOLATED_PAD, group[0] + ISOLATED_PAD
        else:
            left_ext = min(EDGE_PAD_MAX, group[1] - group[0])
            right_ext = min(EDGE_PAD_MAX, group[-1] - group[-2])
            lo, hi = group[0] - left_ext, group[-1] + right_ext
        result.append((lo, hi, binding))
    return result


def render_window(seq, lo, hi, binding_positions):
    """UPPERCASE for a binding residue, lowercase for padding/context, '-'
    for a position with no resolved residue in the structure (a real gap,
    not a numbering artifact) -- see module docstring."""
    chars = []
    for pos in range(lo, hi + 1):
        letter = seq.get(pos)
        if letter is None:
            chars.append("-")
        elif pos in binding_positions:
            chars.append(letter.upper())
        else:
            chars.append(letter.lower())
    return "".join(chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", default=os.path.join(RESULTS_DIR, "aa_type_overlay_assignments.csv"))
    parser.add_argument("--labels-csv", default=os.path.join(RESULTS_DIR, "aa_type_overlay_labels.csv"))
    parser.add_argument("--fp-pickle", default=os.path.join(PROLIF_V2_ROOT, "results",
                                                              "prolif_fingerprint_full_manifest.pkl"))
    parser.add_argument("--structures-dir", default=STRUCTURES_DIR_DEFAULT)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "binding_site_motifs.txt"))
    args = parser.parse_args()

    by_cluster = load_cluster_members(args.assignments)
    labels = load_cluster_labels(args.labels_csv)

    print(f"Loading cached ProLIF fingerprint from {args.fp_pickle} ...")
    ifp_by_site = _load_ifp_by_site(args.fp_pickle)

    seq_cache = {}  # (pdb_id, chain_id) -> {resnum: letter}, shared across sites/clusters
    n_sites, n_skipped = 0, 0
    t0 = time.time()

    cluster_ids_sorted = sorted(by_cluster, key=lambda cid: -len(by_cluster[cid]))
    with open(args.out, "w") as out_f:
        for cluster_id in cluster_ids_sorted:
            members = by_cluster[cluster_id]
            label = labels.get(cluster_id, f"cluster_{cluster_id}")
            out_f.write("=" * 80 + "\n")
            out_f.write(f"POCKET TYPE: {label}  ({len(members)} sites)\n")
            out_f.write("=" * 80 + "\n\n")

            for row in members:
                sid = row["site_id"]
                pdb_id = row.get("pdb_id", "")
                ifp = ifp_by_site.get(sid)
                if not ifp:
                    out_f.write(f">{sid}  [{pdb_id}] -- no cached fingerprint, skipped\n\n")
                    n_skipped += 1
                    continue

                by_chain = binding_residues_by_chain(ifp, DEFAULT_EXCLUDED_INTERACTIONS)
                if not by_chain:
                    out_f.write(f">{sid}  [{pdb_id}] -- no non-metal interacting residues, skipped\n\n")
                    n_skipped += 1
                    continue

                struct_path = find_structure_path(pdb_id, args.structures_dir)
                if struct_path is None:
                    out_f.write(f">{sid}  [{pdb_id}] -- structure file not found, skipped\n\n")
                    n_skipped += 1
                    continue

                out_f.write(f">{sid}  [{pdb_id}]\n")
                for chain_id, positions in sorted(by_chain.items()):
                    cache_key = (pdb_id, chain_id)
                    if cache_key not in seq_cache:
                        seq_cache[cache_key] = load_chain_sequence(struct_path, chain_id)
                    seq = seq_cache[cache_key]
                    if not seq:
                        out_f.write(f"  chain {chain_id}: no sequence extracted (chain not found in structure)\n")
                        continue
                    chain_lo, chain_hi = min(seq), max(seq)
                    for lo, hi, binding in group_and_pad(positions):
                        lo, hi = max(lo, chain_lo), min(hi, chain_hi)
                        window = render_window(seq, lo, hi, binding)
                        out_f.write(f"  chain {chain_id} {lo}-{hi}: {window}\n")
                out_f.write("\n")

                n_sites += 1
                if n_sites % 100 == 0:
                    print(f"  [{n_sites}] sites written ({time.time()-t0:.0f}s elapsed)")

    print(f"\nDone: {n_sites} sites written, {n_skipped} skipped, in {time.time()-t0:.0f}s")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
