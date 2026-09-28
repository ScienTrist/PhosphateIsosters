"""
Step 1b of the pocket_types subproject: collapses reference sites that come
from the same UniProt ID (same protein, e.g. multiple PDB depositions of the
same kinase, or different chains of the same structure) down to one
representative per UniProt group, before re-running clustering.

Why: cluster_pocket_types.py's original run clustered all 658 reference sites
directly. If many of those sites are near-duplicate homologs (multiple PDB
entries of the same or closely related protein), a cluster's SIZE partly
reflects "how many similar structures happen to be in the PDB" rather than
"how many independently-evolved instances of this recognition strategy
exist" -- exactly the redundancy a strict/representative-set clustering step
is meant to remove (see the phosphate-binding-site literature's own
all-against-all + strict-clustering-to-representatives approach this
mirrors).

UniProt grouping is NOT recomputed here -- it already exists from the
project's original RCSB/homology search stage: results/ligand_search_..._json
's "grouped_by_uniprot" (EC class -> UniProt ID -> {"pdbs": [...]}), built
long before ProLIF_v2/pocket_types existed. This script just inverts that
into pdb_id -> uniprot_id and uses it to group reference sites.

Representative choice per UniProt group: the site with the most total
interaction hits (build_pocket_features.py's per-site count) -- the
"best-resolved pocket" convention build_cluster_overlays.pick_anchor also
used to use, before it switched to picking by best average fit to the rest
of its sample instead (total_hits alone doesn't say whether a site is
geometrically typical, which matters there but not for this dedup step).

Sites whose PDB isn't found in the UniProt grouping (a handful expected --
that JSON was built from a specific quality-filtered RCSB query, not
guaranteed to be a superset of every PDB in the current manifest) are kept
as their own singleton group rather than dropped, since "unmapped" isn't
evidence of redundancy.

Usage: python dedup_by_uniprot.py [--long-csv path] [--summary-csv path]
                                   [--ligand-search-json path] [--out-dir path]
"""
import argparse
import csv
import json
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POCKET_TYPES_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(POCKET_TYPES_ROOT)
RESULTS_DIR = os.path.join(POCKET_TYPES_ROOT, "results")

LIGAND_SEARCH_JSON_DEFAULT = os.path.join(
    PROJECT_ROOT, "results", "ligand_search_3D_no_2.7_0.25_Xray_16_0.5_0.8_7.5_0.95_0.8.json")


def build_pdb_to_uniprot(ligand_search_json):
    """grouped_by_uniprot is nested THREE levels deep: EC class -> EC
    subclass (e.g. "1.1") -> UniProt ID -> {"pdbs": [...]} -- confirmed
    directly against the actual JSON (a naive two-level read silently treats
    each EC subclass string as if it were a UniProt ID and crashes on the
    first real lookup)."""
    with open(ligand_search_json) as f:
        data = json.load(f)
    pdb_to_uniprot = {}
    n_collisions = 0
    for ec_class, subclasses in data["grouped_by_uniprot"].items():
        for ec_subclass, uniprot_map in subclasses.items():
            for uniprot_id, info in uniprot_map.items():
                for pdb_id in info["pdbs"]:
                    pdb_id = pdb_id.upper()
                    if pdb_id in pdb_to_uniprot and pdb_to_uniprot[pdb_id] != uniprot_id:
                        n_collisions += 1
                        continue  # keep first mapping seen -- rare multi-EC-class PDB
                    pdb_to_uniprot[pdb_id] = uniprot_id
    print(f"Built pdb_id -> uniprot_id map: {len(pdb_to_uniprot)} PDBs, "
          f"{n_collisions} PDBs seen under >1 UniProt ID (kept first, not merged)")
    return pdb_to_uniprot


def group_and_pick_representatives(summary_rows, pdb_to_uniprot):
    def total_hits(row):
        return (int(row["n_backbone_hits"]) + int(row["n_sidechain_hits"])
                + int(row["n_mixed_hits"]) + int(row["n_metal_hits"]))

    groups = defaultdict(list)
    for row in summary_rows:
        uniprot_id = pdb_to_uniprot.get(row["pdb_id"].upper())
        group_key = uniprot_id if uniprot_id is not None else f"UNMAPPED_{row['site_id']}"
        groups[group_key].append(row)

    representatives = []
    dedup_log = []
    for group_key, rows in groups.items():
        rep = max(rows, key=total_hits)
        representatives.append(rep)
        dedup_log.append({
            "group_key": group_key, "group_size": len(rows),
            "representative_site_id": rep["site_id"], "representative_pdb_id": rep["pdb_id"],
            "member_site_ids": ";".join(r["site_id"] for r in rows),
        })
    return representatives, dedup_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-csv", default=os.path.join(RESULTS_DIR, "ref_site_pocket_bits_long.csv"))
    parser.add_argument("--summary-csv", default=os.path.join(RESULTS_DIR, "ref_site_summary.csv"))
    parser.add_argument("--ligand-search-json", default=LIGAND_SEARCH_JSON_DEFAULT)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    pdb_to_uniprot = build_pdb_to_uniprot(args.ligand_search_json)

    with open(args.summary_csv, newline="") as f:
        summary_rows = list(csv.DictReader(f))
    print(f"{len(summary_rows)} reference sites in {args.summary_csv}")

    n_unmapped = sum(1 for r in summary_rows if r["pdb_id"].upper() not in pdb_to_uniprot)
    print(f"{n_unmapped}/{len(summary_rows)} sites' PDB not found in the UniProt grouping "
          f"(kept as their own singleton group)")

    representatives, dedup_log = group_and_pick_representatives(summary_rows, pdb_to_uniprot)
    rep_site_ids = {r["site_id"] for r in representatives}
    n_groups_with_dupes = sum(1 for row in dedup_log if row["group_size"] > 1)
    n_sites_removed = len(summary_rows) - len(representatives)
    print(f"{len(summary_rows)} sites -> {len(representatives)} UniProt-deduplicated representatives "
          f"({n_sites_removed} removed as duplicates, {n_groups_with_dupes} groups had >1 site)")

    dedup_log_path = os.path.join(args.out_dir, "dedup_by_uniprot_log.csv")
    with open(dedup_log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group_key", "group_size", "representative_site_id",
                                           "representative_pdb_id", "member_site_ids"])
        w.writeheader()
        w.writerows(sorted(dedup_log, key=lambda r: -r["group_size"]))
    print(f"Wrote {dedup_log_path}")

    summary_out_path = os.path.join(args.out_dir, "ref_site_summary_deduped.csv")
    with open(summary_out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(r for r in summary_rows if r["site_id"] in rep_site_ids)
    print(f"Wrote {summary_out_path}")

    with open(args.long_csv, newline="") as f:
        long_rows = list(csv.DictReader(f))
    long_out_path = os.path.join(args.out_dir, "ref_site_pocket_bits_long_deduped.csv")
    with open(long_out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(r for r in long_rows if r["site_id"] in rep_site_ids)
    print(f"Wrote {long_out_path}")


if __name__ == "__main__":
    main()
