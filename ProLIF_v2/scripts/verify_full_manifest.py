"""
Re-runs compare_with_homebrew_plif.py's own score_pair() against every row
already published in plif_vs_tanimoto_comparison_full_manifest.csv and diffs
the result against the published plif_score column.

Context: a bug was found (and fixed) in explain_pair.py -- a SEPARATE, new
diagnostic script that reads from ProLIF_v2/data/pockets/ (native,
unaligned frame) for the PyMOL browser's per-interaction breakdown feature.
compare_with_homebrew_plif.py itself reads from results/motif_analysis/
(pre-aligned at extraction time) and was never touched. This script proves
that directly: if every row here reproduces its published score, the
plif_vs_tanimoto_comparison_full_manifest.csv backing the correlation
artifact was correct all along and does not need to be regenerated.

Usage: python verify_full_manifest.py [manifest.csv]
"""
import csv
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import compare_with_homebrew_plif as cwh  # noqa: E402

PROLIF_V2_ROOT = os.path.dirname(SCRIPT_DIR)
MANIFEST_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROLIF_V2_ROOT, "full_manifest.csv")
_tag = os.path.splitext(os.path.basename(MANIFEST_PATH))[0]
_suffix = "" if _tag == "sample_manifest" else f"_{_tag}"
COMPARISON_CSV = os.path.join(PROLIF_V2_ROOT, "results", f"plif_vs_tanimoto_comparison{_suffix}.csv")


def main():
    with open(MANIFEST_PATH) as f:
        manifest = {r["site_id"]: r for r in csv.DictReader(f)}
    with open(COMPARISON_CSV) as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    n_ok = n_mismatch = n_skip = n_fail = 0
    mismatches = []
    t0 = time.time()

    for i, row in enumerate(rows, 1):
        if row["plif_score"] == "":
            n_skip += 1
            continue
        ref_row = manifest.get(row["ref_site"])
        hit_row = manifest.get(row["hit_site"])
        if ref_row is None or hit_row is None:
            n_skip += 1
            continue

        try:
            result, err = cwh.score_pair(ref_row, hit_row)
        except Exception as e:
            result, err = None, f"{type(e).__name__}: {e}"

        if result is None:
            n_fail += 1
            mismatches.append((row["ref_site"], row["hit_site"], row["plif_score"], None, err))
            continue

        published = float(row["plif_score"])
        recomputed = result["score"]
        if abs(published - recomputed) > 1e-6:
            n_mismatch += 1
            mismatches.append((row["ref_site"], row["hit_site"], published, recomputed, None))
        else:
            n_ok += 1

        if i % 500 == 0 or i == total:
            elapsed = time.time() - t0
            eta = elapsed / i * (total - i)
            print(f"  [{i}/{total}]  ok={n_ok} mismatch={n_mismatch} fail={n_fail} skip={n_skip}  "
                  f"(elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)", flush=True)

    print(f"\n=== Done: {n_ok} ok, {n_mismatch} mismatch, {n_fail} fail, {n_skip} skipped (of {total}) ===")
    if mismatches:
        print(f"\nFirst {min(20, len(mismatches))} mismatches:")
        for ref_site, hit_site, published, recomputed, err in mismatches[:20]:
            if err:
                print(f"  {ref_site} vs {hit_site}: FAILED to recompute ({err}), published={published}")
            else:
                print(f"  {ref_site} vs {hit_site}: published={published:.4f} recomputed={recomputed:.4f}")


if __name__ == "__main__":
    main()
