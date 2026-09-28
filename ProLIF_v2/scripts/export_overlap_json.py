"""
Converts results/plif_vs_tanimoto_comparison_full_manifest.csv into a compact JSON
array for embedding in the homebrew-vs-ProLIF overlap dashboard artifact (short keys
to keep the embedded payload small; see the dashboard's own JS for the key meanings).

Run again any time the comparison CSV is regenerated (e.g. after a run_prolif.py +
compare_with_homebrew_plif.py rerun) to refresh the dashboard's data.
"""
import csv
import json
import os
import sys

PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_TAG = sys.argv[1] if len(sys.argv) > 1 else "full_manifest"
IN_CSV = os.path.join(PROLIF_V2_ROOT, "results", f"plif_vs_tanimoto_comparison_{MANIFEST_TAG}.csv")
OUT_JSON = os.path.join(PROLIF_V2_ROOT, "results", f"overlap_dashboard_data_{MANIFEST_TAG}.json")


def to_float(v):
    return float(v) if v not in (None, "") else None


def to_int(v):
    return int(v) if v not in (None, "") else None


def split_site(site_id):
    # site_id format: ref_{pdbid}_{ligresname}_{resnum} or hit_{pdbid}_{ligresname}_{resnum}
    parts = site_id.split("_")
    pdbid, resnum = parts[1], parts[-1]
    lig = "_".join(parts[2:-1])
    return pdbid, lig, resnum


def main():
    with open(IN_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    has_prolif_ref_n = rows and "prolif_ref_n" in rows[0]
    out = []
    for row in rows:
        ref_pdb, ref_lig, ref_num = split_site(row["ref_site"])
        hit_pdb, hit_lig, hit_num = split_site(row["hit_site"])
        out.append({
            "rs": row["ref_site"], "hs": row["hit_site"],
            "rp": ref_pdb, "rl": ref_lig, "rn": ref_num,
            "hp": hit_pdb, "hl": hit_lig, "hn": hit_num,
            "hbs": to_float(row["plif_score"]),
            "hbn": to_int(row["plif_ref_n"]),
            "pfs": to_float(row["prolif_plif_score"]),
            "pfn": to_int(row["prolif_ref_n"]) if has_prolif_ref_n else None,
            "nca": to_int(row["n_ca_mapped"]),
            "sg": to_int(row["same_phosphate_group"]),
        })

    with open(OUT_JSON, "w") as f:
        json.dump({"has_prolif_ref_n": has_prolif_ref_n, "rows": out}, f, separators=(",", ":"))

    print(f"Wrote {OUT_JSON} ({len(out)} pairs, prolif_ref_n {'present' if has_prolif_ref_n else 'MISSING -- rerun run_prolif.py/compare_with_homebrew_plif.py first'})")


if __name__ == "__main__":
    main()
