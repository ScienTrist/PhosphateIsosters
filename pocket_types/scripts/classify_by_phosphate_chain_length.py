"""
Adjunct to classify_by_aa_type.py: labels each reference site by how many
phosphate groups its ligand has (mono/di/tri-phosphate, by P-atom count --
AMP=1, ADP=2, ATP=3, etc.), then cross-tabulates against the AA-type
categories already computed. This is a real, previously-ignored confound:
classify_by_aa_type.py's categories were built purely from interacting-
residue composition, with no regard for whether the ligand itself is a
mono/di/tri-phosphate chain -- two sites landing in the same "ARG:1" category
could be recognizing chemically different numbers of phosphates.

Phosphate-group count reuses phosphate_ifp.get_ligand_phosphate_groups()
(the same helper phosphate_ifp.py/geometric_subcluster.py already use to
find the ligand's phosphate group(s)) -- one entry per phosphorus atom, each
with its own coordinating O/N within 2.1A. This is a proxy for "how many
phosphates", not a strict verification that they're chain-bonded to each
other (two independent monophosphate groups on the same ligand would also
count as 2) -- reasonable for this project's ligand population (nucleotides/
nucleotide analogs dominate), but not a chemical connectivity check.

Only needs the LIGAND mol (not the protein), so this is much cheaper per
site than build_pocket_features.py/classify_by_aa_type.py's own per-site work.

Usage: python classify_by_phosphate_chain_length.py [--manifest path]
                                                      [--categories-csv path]
                                                      [--pockets-dir path] [--out-dir path]
"""
import argparse
import csv
import os
import sys
import time
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POCKET_TYPES_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(POCKET_TYPES_ROOT)
PROLIF_V2_ROOT = os.path.join(PROJECT_ROOT, "ProLIF_v2")
RESULTS_DIR = os.path.join(POCKET_TYPES_ROOT, "results")

sys.path.insert(0, os.path.join(PROLIF_V2_ROOT, "scripts"))
import run_prolif as rp  # noqa: E402
import phosphate_ifp as pif  # noqa: E402
import MDAnalysis as mda  # noqa: E402

sys.path.insert(0, SCRIPT_DIR)
from build_pocket_features import _manifest_reference_sites  # noqa: E402

CHAIN_LABELS = {1: "mono", 2: "di", 3: "tri"}


def n_phosphates_for_site(site_id, manifest_row, pockets_dir):
    """Returns the ligand's phosphorus-atom count (== number of phosphate
    groups), or None if the pocket/ligand can't be loaded."""
    pocket_path = os.path.join(pockets_dir, f"{site_id}.pdb")
    if not os.path.exists(pocket_path):
        return None
    u = mda.Universe(pocket_path)
    ligand_ag = u.select_atoms(
        f"resid {manifest_row['lig_resnum']} and chainID {manifest_row['chain']} and not protein")
    if len(ligand_ag) == 0:
        return None
    lig_mol = rp.safe_molecule_from_mda(ligand_ag, use_segid=False, fix_ionizable=True)
    groups = pif.get_ligand_phosphate_groups(lig_mol)
    return len(groups)


def chain_label(n):
    return CHAIN_LABELS.get(n, f"{n}-phosphate" if n else "none")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join(PROLIF_V2_ROOT, "full_manifest.csv"))
    parser.add_argument("--categories-csv", default=os.path.join(RESULTS_DIR, "aa_type_categories.csv"))
    parser.add_argument("--pockets-dir", default=os.path.join(PROLIF_V2_ROOT, "data", "pockets"))
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    ref_sites = _manifest_reference_sites(args.manifest)
    print(f"{len(ref_sites)} unique reference sites in {args.manifest}")

    rows = []
    n_fail = 0
    t0 = time.time()
    for i, (site_id, site) in enumerate(sorted(ref_sites.items()), 1):
        n = n_phosphates_for_site(site_id, site, args.pockets_dir)
        if n is None:
            n_fail += 1
            continue
        rows.append({"site_id": site_id, "pdb_id": site["pdb_id"], "ref_ligand": site["lig_resname"],
                      "n_phosphates": n, "chain_label": chain_label(n)})
        if i % 150 == 0 or i == len(ref_sites):
            print(f"  [{i}/{len(ref_sites)}] {len(rows)} ok, {n_fail} failed ({time.time()-t0:.0f}s elapsed)")

    print(f"\n{len(rows)}/{len(ref_sites)} sites classified ({n_fail} failed to load)")

    label_counts = Counter(r["chain_label"] for r in rows)
    print("Overall phosphate-chain-length distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label:10s} {count}")

    chain_path = os.path.join(args.out_dir, "phosphate_chain_length.csv")
    with open(chain_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site_id", "pdb_id", "ref_ligand", "n_phosphates", "chain_label"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {chain_path}")

    # Cross-tabulate against classify_by_aa_type.py's categories, if available.
    if not os.path.exists(args.categories_csv):
        print(f"{args.categories_csv} not found -- skipping cross-tabulation")
        return

    chain_by_site = {r["site_id"]: r["chain_label"] for r in rows}
    category_by_site = {}
    with open(args.categories_csv, newline="") as f:
        for r in csv.DictReader(f):
            category_by_site[r["site_id"]] = r["category"]

    cross = defaultdict(Counter)
    category_sizes = Counter()
    for site_id, category in category_by_site.items():
        category_sizes[category] += 1
        label = chain_by_site.get(site_id)
        if label:
            cross[category][label] += 1

    cross_path = os.path.join(args.out_dir, "aa_type_by_phosphate_chain_length.txt")
    with open(cross_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("AA-type categories (classify_by_aa_type.py) crossed with phosphate chain\n")
        f.write("length (mono/di/tri, by ligand P-atom count)\n")
        f.write("=" * 80 + "\n\n")
        for category, size in sorted(category_sizes.items(), key=lambda kv: -kv[1]):
            counts = cross[category]
            breakdown = ", ".join(f"{label}={n}" for label, n in counts.most_common())
            f.write(f"{category:20s}  {size:4d} sites  --  {breakdown}\n")
    print(f"Wrote {cross_path}")


if __name__ == "__main__":
    main()
