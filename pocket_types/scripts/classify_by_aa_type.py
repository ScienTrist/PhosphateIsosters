"""
Step 5 of pocket_types (alternative to cluster_pocket_types.py + geometric_
subcluster.py): a deterministic, rule-based classification of reference
sites by the exact multiset of non-backbone (sidechain + metal) interacting
residue types, instead of any distance-based clustering algorithm.

Why: both the composition-vector KMeans clustering AND the geometric RMSD/
Louvain sub-clustering kept converging on only ~4 top-level groups, with the
Arg/Lys salt-bridge population either forming one dominant blob or getting
arbitrarily fragmented by Louvain's modularity optimization (confirmed
directly: three separate runs of geometric_subcluster.py, at different point
representations and thresholds, all either merged chemically distinct types
together or split one chemically uniform type into 2-3 pieces). Two real
papers doing similar phosphate-binding-site classification both report
10-20 distinct types. A rule-based categorical key sidesteps every
algorithm-tuning problem that produced this instability (no RMSD threshold,
no Louvain resolution limit, no k selection) at the cost of geometric
precision -- it groups by WHICH residue types+counts are present, not by 3D
arrangement.

Category key: sorted tuple of (resname, distinct_residue_count) for every
DISTINCT residue (by chain+number, not raw interaction-instance count -- one
Arg forming two separate H-bonds is still "1 Arg", not "2") with a non-
backbone (sidechain or metal; "mixed" folded in here too, since by
definition it includes some non-backbone contribution) qualifying
interaction -- e.g. one Lys + one Arg salt bridge keys as
"ARG:1,LYS:1", independent of exactly which residues or their 3D
arrangement. Backbone contacts are NOT broken out by residue identity
(backbone amide H-bonding is largely residue-identity-independent
chemistry, unlike a side chain's specific functional group, so which
residue happens to contribute a backbone NH/C=O is mostly a loop-geometry
accident, not a chemical choice worth splitting categories on) -- but
whether a site uses backbone H-bonding AT ALL is still a real chemical
axis (a pocket that supplements a salt bridge with a backbone amide is
subtly different from one that doesn't), so it's folded into the key as a
flat "+BB" suffix, e.g. "LYS:1" vs "LYS:1+BB". A site with ONLY backbone
contacts (no sidechain/metal at all) gets an "NBB" category instead of an
empty key, where N is the distinct-residue COUNT of backbone-contributing
residues -- e.g. "1BB" (one backbone H-bond), "2BB" (two), etc. -- since
backbone-only sites can still differ by how many separate backbone contacts
they make, even though (per above) it's the count that matters, not which
residues. has_backbone is still recorded as its own boolean column for
convenience.

A site can also have ZERO qualifying interactions of any kind -- every
protein residue near its phosphate only registered VdWContact (excluded by
excluded_interactions), so residues_by_type ends up completely empty. This
used to get silently folded into the same "BACKBONE_ONLY" bucket as genuine
backbone-H-bond sites, which was wrong: confirmed directly against the raw
fingerprint pickle, e.g. ref_2JB7_AMP_1168's only recorded interaction is
"SER:VdWContact" -- no backbone H-bond, no sidechain/metal contact, nothing
qualifying at all. These now get their own explicit "NO_INT" category,
distinct from "1BB"/"2BB"/etc.

All metal elements (MG/ZN/MN/CA/FE/CU/CO/NI) are collapsed into one
"METAL" bucket before counting -- a phosphate-coordinating divalent cation
plays the same structural role regardless of which element it is, so e.g.
one Lys + one Mg keys as "LYS:1,METAL:1", same as one Lys + one Zn would.

Residue identity/provenance is read straight from the cached ProLIF
fingerprint pickle + a per-site protein-mol rebuild, the same source (and
same _provenance() backbone/sidechain/metal classifier, including its
Ca2+/alpha-carbon name-collision fix) build_pocket_features.py/geometric_
subcluster.py already use -- NOT the aggregated ref_site_pocket_bits_long.csv,
whose "count" column is interaction-INSTANCE count, not distinct-residue
count, and can't be reverse-engineered into the latter.

Usage: python classify_by_aa_type.py [--manifest path] [--fp-pickle path]
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

sys.path.insert(0, SCRIPT_DIR)
from build_pocket_features import (  # noqa: E402
    _load_ifp_by_site, _protein_mol_for_site, _provenance,
    _manifest_reference_sites, DEFAULT_EXCLUDED_INTERACTIONS,
)

NO_INTERACTION_KEY = "NO_INT"
NON_BACKBONE_PROVENANCE = {"sidechain", "metal", "mixed"}
METAL_BUCKET = "METAL"


def site_category(site_id, ifp, prot_mol, excluded_interactions):
    """Returns (key_str, has_backbone) for one site. key_str is one of:
      - NO_INTERACTION_KEY, if literally no qualifying interaction exists
        (every nearby contact was VdWContact or similar, excluded up front)
      - "NBB" (e.g. "1BB", "2BB"), if the only qualifying interactions are
        backbone ones -- N is the distinct-residue count of backbone
        contributors, not broken out by resname (see module docstring)
      - a comma-joined "RESNAME:count" string sorted by resname, e.g.
        "ARG:1,LYS:1", optionally with a "+BB" suffix if backbone also
        contributes alongside the sidechain/metal contacts."""
    # (resname, provenance) -> {(chain, number), ...} -- distinct residues,
    # not raw interaction-instance counts.
    residues_by_type = defaultdict(set)
    for (lig_id, prot_id), interactions in ifp.items():
        qualifying_names = [name for name in interactions if name not in excluded_interactions]
        if not qualifying_names:
            continue
        atom_idxs = [idx for name in qualifying_names for md in interactions[name]
                     for idx in md["parent_indices"]["protein"]]
        provenance = _provenance(prot_mol, atom_idxs, prot_id.name)
        # All metal elements (Mg/Zn/Mn/Ca/Fe/Cu/Co/Ni) are chemically
        # interchangeable for this project's purposes -- a phosphate cares
        # that a divalent cation is bridging it, not which element it is --
        # so they're bucketed into one METAL_BUCKET resname before counting,
        # same as they already share one "metal" provenance.
        key_resname = METAL_BUCKET if provenance == "metal" else prot_id.name
        residues_by_type[(key_resname, provenance)].add((prot_id.chain, prot_id.number))

    if not residues_by_type:
        return NO_INTERACTION_KEY, False

    has_backbone = any(prov == "backbone" for _, prov in residues_by_type)
    non_backbone_counts = Counter()
    n_backbone_residues = 0
    for (resname, provenance), residue_ids in residues_by_type.items():
        if provenance in NON_BACKBONE_PROVENANCE:
            non_backbone_counts[resname] += len(residue_ids)
        elif provenance == "backbone":
            n_backbone_residues += len(residue_ids)

    if not non_backbone_counts:
        return f"{n_backbone_residues}BB", has_backbone
    key_str = ",".join(f"{resname}:{count}" for resname, count in sorted(non_backbone_counts.items()))
    if has_backbone:
        key_str += "+BB"
    return key_str, has_backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join(PROLIF_V2_ROOT, "full_manifest.csv"))
    parser.add_argument("--fp-pickle", default=os.path.join(PROLIF_V2_ROOT, "results",
                                                              "prolif_fingerprint_full_manifest.pkl"))
    parser.add_argument("--pockets-dir", default=os.path.join(PROLIF_V2_ROOT, "data", "pockets"))
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    ref_sites = _manifest_reference_sites(args.manifest)
    print(f"{len(ref_sites)} unique reference sites in {args.manifest}")

    print(f"Loading cached ProLIF fingerprint from {args.fp_pickle} ...")
    ifp_by_site = _load_ifp_by_site(args.fp_pickle)

    rows = []
    n_fail = 0
    t0 = time.time()
    for i, (site_id, site) in enumerate(sorted(ref_sites.items()), 1):
        ifp = ifp_by_site.get(site_id)
        if not ifp:
            n_fail += 1
            continue
        prot_mol, err = _protein_mol_for_site(site_id)
        if prot_mol is None:
            n_fail += 1
            continue
        key_str, has_backbone = site_category(site_id, ifp, prot_mol, DEFAULT_EXCLUDED_INTERACTIONS)
        rows.append({
            "site_id": site_id, "pdb_id": site["pdb_id"], "ref_ligand": site["lig_resname"],
            "category": key_str, "has_backbone": has_backbone,
        })
        if i % 100 == 0 or i == len(ref_sites):
            print(f"  [{i}/{len(ref_sites)}] {len(rows)} ok, {n_fail} failed ({time.time()-t0:.0f}s elapsed)")

    print(f"\n{len(rows)}/{len(ref_sites)} sites classified ({n_fail} failed to load)")

    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    print(f"{len(by_category)} distinct categories")

    assignments_path = os.path.join(args.out_dir, "aa_type_categories.csv")
    with open(assignments_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site_id", "pdb_id", "ref_ligand", "category", "has_backbone"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {assignments_path}")

    summary_path = os.path.join(args.out_dir, "aa_type_categories_summary.txt")
    ranked = sorted(by_category.items(), key=lambda kv: -len(kv[1]))
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Reference sites classified by sidechain/metal interacting residue type+\n")
        f.write(f"count, plus a '+BB' suffix when backbone H-bonding also contributes.\n")
        f.write(f"Backbone-only sites are 'NBB' (N = distinct backbone-contact count);\n")
        f.write(f"sites with no qualifying interaction at all are 'NO_INT' -- \n")
        f.write(f"{len(rows)} sites, {len(by_category)} categories\n")
        f.write("=" * 80 + "\n\n")
        for category, members in ranked:
            f.write(f"{category}  --  {len(members)} sites\n")
            for m in sorted(members, key=lambda r: r["site_id"])[:6]:
                f.write(f"    {m['site_id']:28s} {m['pdb_id']:5s} {m['ref_ligand']:5s}\n")
            f.write("\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
