"""
One-off comparison: pick pairs that already have a PLIF score in
results/motif_analysis/plif_results.json, run them through the new real-PLIP
Tanimoto pipeline, and print both scores side by side.
"""
import json
import sys

from common import PROJECT_ROOT, get_transformation
from resolve_pairs import resolve_row
from score_pairs import score_pair

PLIF_RESULTS = f"{PROJECT_ROOT}/results/motif_analysis/plif_results.json"


def main(n=40):
    plif = json.load(open(PLIF_RESULTS))
    plif_sorted = sorted(plif, key=lambda r: -r["score"])
    step = max(1, len(plif_sorted) // n)
    sample = plif_sorted[::step][:n]

    print(f"{'Ref':<6}{'Hit':<6}{'Mimic':<8}{'PLIF':>7}{'Tanimoto':>10}{'n_ref':>7}{'n_hit':>7}{'n_match':>8}  note")
    rows_out = []
    for r in sample:
        fake_row = {
            "Ref_ID": r["ref"], "Hit_ID": r["hit"],
            "Ref_Lig": r["ref_lig"], "Ref_Num": r["ref_num"],
            "Hit_Ligand": r["mimic"],
            "Distance": "", "EC_Class": r["ec"], "Metal_Status": "", "State": "HOLO",
        }
        resolved, err = resolve_row(fake_row)
        if resolved is None:
            print(f"{r['ref']:<6}{r['hit']:<6}{r['mimic']:<8}{r['score']:>7.3f}{'--':>10}{'':>7}{'':>7}{'':>8}  [resolve failed] {err}")
            continue

        result, err2 = score_pair(resolved)
        if result is None:
            print(f"{r['ref']:<6}{r['hit']:<6}{r['mimic']:<8}{r['score']:>7.3f}{'--':>10}{'':>7}{'':>7}{'':>8}  [score failed] {err2}")
            continue

        print(f"{r['ref']:<6}{r['hit']:<6}{r['mimic']:<8}{r['score']:>7.3f}{result['tanimoto']:>10.4f}"
              f"{result['n_ref']:>7}{result['n_hit']:>7}{result['n_matched']:>8}")
        rows_out.append((r["ref"], r["hit"], r["mimic"], r["score"], result["tanimoto"]))

    if rows_out:
        import statistics
        plif_vals = [x[3] for x in rows_out]
        tan_vals = [x[4] for x in rows_out]
        if len(rows_out) > 1:
            r = statistics.correlation(plif_vals, tan_vals)
            print(f"\nn={len(rows_out)} comparable pairs, Pearson r = {r:.3f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    main(n)
