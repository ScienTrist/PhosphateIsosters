"""
Step 7: computes the homebrew PLIF spatial-similarity score (scripts/analyze_plif.py's
get_interactions + calculate_spatial_similarity) for the exact same ref/mimic ligand
pair used in each manifest row, then merges it with run_prolif.py's three Tanimoto
variants (results/prolif_pair_similarity.csv) into one comparison table.

Scored directly against the manifest's specific (ref_lig, hit_lig) pair rather than
reusing results/motif_analysis/plif_results.json, because that file only records the
greedy-assigned *best-scoring* mimic per reference phosphate group -- which is not
necessarily the same ligand instance select_sample.py/resolve_pairs.py picked (a
nearest-copy spatial match, independent of PLIF score). Scoring the manifest's own
pair directly keeps this an apples-to-apples comparison: same ligand instance, two
different scoring methods.
"""
import csv
import glob
import os
import sys
import time

import gemmi


PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PROLIF_V2_ROOT)
# CLI override, e.g. `python compare_with_homebrew_plif.py full_manifest.csv` -- same
# pattern run_prolif.py now uses. Must match whatever manifest run_prolif.py was
# itself run against, since TANIMOTO_CSV below has to be the matching
# prolif_pair_similarity file that run produced, not a different manifest's.
MANIFEST_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROLIF_V2_ROOT, "sample_manifest.csv")
_manifest_tag = os.path.splitext(os.path.basename(MANIFEST_PATH))[0]
RUN_SUFFIX = "" if _manifest_tag == "sample_manifest" else f"_{_manifest_tag}"
TANIMOTO_CSV = os.path.join(PROLIF_V2_ROOT, "results", f"prolif_pair_similarity{RUN_SUFFIX}.csv")
OUT_CSV = os.path.join(PROLIF_V2_ROOT, "results", f"plif_vs_tanimoto_comparison{RUN_SUFFIX}.csv")

MOTIF_DIR = os.path.join(PROJECT_ROOT, "results", "motif_analysis")
REFS_DIR = os.path.join(MOTIF_DIR, "references")
HITS_DIR = os.path.join(MOTIF_DIR, "hits")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from analyze_plif import get_interactions, get_interactions_per_phosphate, get_metal_positions, calculate_spatial_similarity  # noqa: E402
from utils import fmt_eta  # noqa: E402


def find_ref_path(ref_id, chain, lig, num):
    matches = glob.glob(os.path.join(REFS_DIR, f"ref_{ref_id}_{chain}_{lig}_{num}.cif"))
    return matches[0] if matches else None


def find_hit_path(ref_id, hit_id, chain, ref_lig, ref_num):
    matches = glob.glob(os.path.join(HITS_DIR, "**", f"hit_{ref_id}_{hit_id}_{chain}_{ref_lig}_{ref_num}.cif"),
                         recursive=True)
    return matches[0] if matches else None


def score_pair(ref_row, hit_row):
    ref_path = find_ref_path(ref_row["pdb_id"], ref_row["chain"], ref_row["lig_resname"], ref_row["lig_resnum"])
    hit_path = find_hit_path(ref_row["pdb_id"], hit_row["pdb_id"], ref_row["chain"],
                              ref_row["lig_resname"], ref_row["lig_resnum"])
    if ref_path is None or hit_path is None:
        return None, f"missing motif_analysis file (ref={ref_path is not None}, hit={hit_path is not None})"

    st_ref = gemmi.read_structure(ref_path)
    st_hit = gemmi.read_structure(hit_path)

    # Reference is scored on its phosphate group only (not the whole ligand) --
    # for ligands with multiple phosphates (e.g. ATP), take the group with the
    # most protein contacts. The hit has no known phosphate to anchor on, so it
    # stays whole-ligand: any part of the mimic could be the part that matters.
    #
    # This is a DIFFERENT, hit-independent selection than run_prolif.py's own
    # (which picks whichever group has the most matched interactions against
    # THIS specific hit, per pair -- see its "same_phosphate_group" comment in
    # main() below). Deliberately not unified: this project treats the two
    # scores as independent methods to compare, not one feeding the other, so
    # homebrew keeps its own busiest-group-only convention rather than
    # adopting ProLIF's per-hit choice. homebrew_p_idx is still captured here
    # (previously discarded by max() -- only the winning group's interactions
    # were kept, not which p_idx won) purely so main() can flag whenever the
    # two methods happened to land on different groups for a given pair.
    ref_inters_per_p = get_interactions_per_phosphate(st_ref, ref_row["lig_resname"], ref_row["lig_resnum"])
    if ref_inters_per_p:
        homebrew_p_idx, ref_inters = max(ref_inters_per_p.items(), key=lambda kv: len(kv[1]))
    else:
        homebrew_p_idx, ref_inters = None, []
    hit_inters = get_interactions(st_hit, hit_row["lig_resname"], hit_row["lig_resnum"])
    # calculate_spatial_similarity returns 0.0 (not an error) when ref_inters is empty --
    # keep that pair in the output as score=0.0 rather than dropping it. Only a missing
    # structure file (checked above) is a real failure worth excluding a row for.

    hit_metals = get_metal_positions(st_hit)
    mimic_positions = []
    for chain in st_hit[0]:
        for res in chain:
            if res.name == hit_row["lig_resname"] and str(res.seqid.num) == str(hit_row["lig_resnum"]):
                mimic_positions = [(a.pos.x, a.pos.y, a.pos.z) for a in res]

    score, _ = calculate_spatial_similarity(ref_inters, hit_inters, hit_metals, mimic_positions)
    return {"score": score, "ref_n": len(ref_inters), "hit_n": len(hit_inters),
            "homebrew_p_idx": homebrew_p_idx}, None


def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    by_site_id = {r["site_id"]: r for r in rows}  # last-seen wins; identical across dupes

    with open(TANIMOTO_CSV) as f:
        tanimoto_rows = list(csv.DictReader(f))

    total = len(tanimoto_rows)
    print(f"Scoring homebrew PLIF for {total} pairs...")
    out_rows = []
    n_ok = n_fail = 0
    t0 = time.time()
    for i, row in enumerate(tanimoto_rows, 1):
        site_a, site_b = row["site_a"], row["site_b"]
        ref_sid, hit_sid = (site_a, site_b) if row["role_a"] == "reference" else (site_b, site_a)
        ref_row, hit_row = by_site_id[ref_sid], by_site_id[hit_sid]

        try:
            result, err = score_pair(ref_row, hit_row)
        except Exception as e:
            result, err = None, f"{type(e).__name__}: {e}"

        if result is None:
            n_fail += 1
        else:
            n_ok += 1
        elapsed = time.time() - t0
        eta = elapsed / i * (total - i) if i else 0
        progress = f"[{i}/{total} {100*i/total:3.0f}%  ok={n_ok} fail={n_fail}]"

        if result is None:
            print(f"  {progress} FAIL {ref_sid} vs {hit_sid}: {err} (ETA {fmt_eta(eta)})")
            continue

        # same_phosphate_group: whether homebrew's (busiest-by-count, hit-independent)
        # and ProLIF's (most-matched-against-this-hit) phosphate group picks agree for
        # this specific pair -- NOT used to drop or reweight anything here, purely a
        # flag for filtering downstream (e.g. restrict a correlation plot to 1-only for
        # a strict apples-to-apples comparison, or look at 0-only to see how much the
        # two methods' group choices actually diverge). See score_pair()'s docstring
        # for why the two selections are kept independent rather than unified.
        #
        # 1 (agree / not applicable) covers three cases: (a) both sides picked the same
        # p_idx, (b) ProLIF's ref_phosphate_p_idx is blank -- it only ever populates
        # this for references it treated as multi-phosphate (>1 group with contacts),
        # so blank means ProLIF itself saw nothing to choose between, and (c) homebrew
        # found no phosphate group at all (homebrew_p_idx is None). 0 (disagree) is
        # the one remaining case: both sides made a real, independent choice, and those
        # choices differ.
        prolif_p_idx = row["ref_phosphate_p_idx"]
        homebrew_p_idx = result["homebrew_p_idx"]
        if prolif_p_idx == "" or homebrew_p_idx is None:
            same_group = 1
        else:
            same_group = 1 if str(homebrew_p_idx) == prolif_p_idx else 0

        print(f"  {progress} ok {ref_row['pdb_id']}/{hit_row['pdb_id']} ({hit_row['lig_resname']}): "
              f"PLIF={result['score']:.3f}  Tanimoto(aligned)={float(row['tanimoto_aligned_residue']) if row['tanimoto_aligned_residue'] else 'n/a'}"
              f"  ProLIF_PLIF={float(row['prolif_plif_score']) if row['prolif_plif_score'] else 'n/a'}"
              f"  same_phosphate_group={same_group} (ETA {fmt_eta(eta)})")
        out_rows.append({
            "ref_site": ref_sid, "hit_site": hit_sid,
            "ref_id": ref_row["pdb_id"], "hit_id": hit_row["pdb_id"], "mimic": hit_row["lig_resname"],
            "plif_score": result["score"], "plif_ref_n": result["ref_n"], "plif_hit_n": result["hit_n"],
            "tanimoto_aligned_residue": row["tanimoto_aligned_residue"],
            "prolif_plif_score": row["prolif_plif_score"],
            "prolif_ref_n": row.get("prolif_ref_n", ""),
            "n_ca_mapped": row["n_ca_mapped"],
            "ref_phosphate_p_idx": row["ref_phosphate_p_idx"],
            "homebrew_phosphate_p_idx": homebrew_p_idx if homebrew_p_idx is not None else "",
            "same_phosphate_group": same_group,
        })

    fieldnames = ["ref_site", "hit_site", "ref_id", "hit_id", "mimic",
                  "plif_score", "plif_ref_n", "plif_hit_n",
                  "tanimoto_aligned_residue", "prolif_plif_score", "prolif_ref_n", "n_ca_mapped",
                  "ref_phosphate_p_idx", "homebrew_phosphate_p_idx", "same_phosphate_group"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n{n_ok}/{len(tanimoto_rows)} pairs scored, {n_fail} failed (missing motif_analysis files or "
          f"zero ref interactions).")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
