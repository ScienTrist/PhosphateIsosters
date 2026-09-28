"""
Builds a ProLIF_v2 manifest covering the ENTIRE resolved candidate pool
(results/plip_isostere/resolved_pairs.csv -- every isostere_full_list.csv row
with a candidate mimic ligand that successfully resolved to an exact chain/
resnum instance, see resolve_pairs.py), not just the 50-pair diverse sample.

Writes to full_manifest.csv (NOT sample_manifest.csv -- that file is the
curated 50-pair phosphate-diversity deliverable and must not be overwritten
by this uncapped run).

Row-building mirrors select_sample.py's site_id convention exactly
(ref_{ID}_{Lig}_{Num} / hit_{ID}_{Lig}_{Num}) so every other script in this
pipeline (extract_pockets.py, run_prolif.py, phosphate_ifp.py,
batch_prepare_structures.py) works against this manifest unchanged.
"""
import csv
import os

PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PROLIF_V2_ROOT)
RESOLVED_PAIRS_CSV = os.path.join(PROJECT_ROOT, "results", "plip_isostere", "resolved_pairs.csv")
FULL_MANIFEST_PATH = os.path.join(PROLIF_V2_ROOT, "full_manifest.csv")


def make_site(role, resolved, paired_with):
    prefix = "ref" if role == "reference" else "hit"
    id_key, lig_key, num_key, chain_key = (
        ("Ref_ID", "Ref_Lig", "Ref_Num", "Ref_Chain") if role == "reference"
        else ("Hit_ID", "Hit_Ligand", "Hit_Num", "Hit_Chain")
    )
    return {
        "site_id": f"{prefix}_{resolved[id_key]}_{resolved[lig_key]}_{resolved[num_key]}",
        "role": role,
        "pdb_id": resolved[id_key],
        "chain": resolved[chain_key],
        "lig_resname": resolved[lig_key],
        "lig_resnum": str(resolved[num_key]),
        "paired_with": paired_with,
    }


def main():
    with open(RESOLVED_PAIRS_CSV) as f:
        resolved_rows = list(csv.DictReader(f))

    manifest = []
    for resolved in resolved_rows:
        manifest.append(make_site("reference", resolved, resolved["Hit_ID"]))
        manifest.append(make_site("hit", resolved, resolved["Ref_ID"]))

    fieldnames = ["site_id", "role", "pdb_id", "chain", "lig_resname", "lig_resnum", "paired_with"]
    with open(FULL_MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest)

    n_ref_sites = len({m["site_id"] for m in manifest if m["role"] == "reference"})
    n_hit_sites = len({m["site_id"] for m in manifest if m["role"] == "hit"})
    print(f"Wrote {FULL_MANIFEST_PATH}: {len(resolved_rows)} pairs ({len(manifest)} rows)")
    print(f"  {n_ref_sites} unique reference sites, {n_hit_sites} unique hit sites")


if __name__ == "__main__":
    main()
