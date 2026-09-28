"""
Step 1: picks 10 rows from the existing isostere_full_list.csv and, for each,
resolves the exact (chain, resname, resnum) of BOTH the reference ligand and
the hit ligand in their own native numbering -- reusing resolve_pairs.resolve_row,
which is already-validated logic for this (TM-align-based nearest-copy matching)
and has no dependency on PLIP.

10 rows -> 20 sites (1 reference + 1 hit each), written to sample_manifest.csv.
Resolution failures (cif-only entries, missing TM-align transform, etc.) are
skipped and replaced by drawing further rows, so the manifest always ends up
with exactly 20 sites regardless of how many candidates fail to resolve.
"""
import csv
import os
import random
import sys

PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PROLIF_V2_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "plip_isostere"))

from common import FULL_LIST_CSV
from resolve_pairs import resolve_row

MANIFEST_PATH = os.path.join(PROLIF_V2_ROOT, "sample_manifest.csv")
N_ROWS = 50
SEED = 42


def main():
    rows = [r for r in csv.DictReader(open(FULL_LIST_CSV)) if r["Hit_Ligand"] not in ("None", "", None)]
    random.Random(SEED).shuffle(rows)

    manifest = []
    n_resolved_rows = 0
    n_tried = 0
    for row in rows:
        if n_resolved_rows >= N_ROWS:
            break
        n_tried += 1
        resolved, err = resolve_row(row)
        if resolved is None:
            print(f"  [skip] {row['Ref_ID']}/{row['Hit_ID']} ({row['Hit_Ligand']}): {err}")
            continue
        manifest.append({
            "site_id": f"ref_{resolved['Ref_ID']}_{resolved['Ref_Lig']}_{resolved['Ref_Num']}",
            "role": "reference",
            "pdb_id": resolved["Ref_ID"],
            "chain": resolved["Ref_Chain"],
            "lig_resname": resolved["Ref_Lig"],
            "lig_resnum": resolved["Ref_Num"],
            "paired_with": resolved["Hit_ID"],
        })
        manifest.append({
            "site_id": f"hit_{resolved['Hit_ID']}_{resolved['Hit_Ligand']}_{resolved['Hit_Num']}",
            "role": "hit",
            "pdb_id": resolved["Hit_ID"],
            "chain": resolved["Hit_Chain"],
            "lig_resname": resolved["Hit_Ligand"],
            "lig_resnum": resolved["Hit_Num"],
            "paired_with": resolved["Ref_ID"],
        })
        n_resolved_rows += 1

    if len(manifest) < N_ROWS * 2:
        raise SystemExit(f"Only resolved {len(manifest)}/{N_ROWS * 2} sites after trying {n_tried} rows "
                          f"-- exhausted the shuffled pool, or too many resolution failures.")

    fieldnames = ["site_id", "role", "pdb_id", "chain", "lig_resname", "lig_resnum", "paired_with"]
    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest)

    print(f"\nResolved {len(manifest)} sites from {n_tried} candidate rows ({SEED=}).")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Unique PDB IDs to download: {len(set(m['pdb_id'] for m in manifest))}")


if __name__ == "__main__":
    main()
