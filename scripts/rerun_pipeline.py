# Re-runs the isostere/PLIF re-classification pipeline in one command:
#   analyze_isosteres_simple -> analyze_plif -> sort_hits -> export_to_datawarrior
#
# Takes a snapshot of apo/holo/plif_confirmed state before and after the run,
# and prints (and saves) a diff report: count changes, newly-confirmed and
# no-longer-confirmed pairs, mimic ligand distribution changes, etc.
#
# Use this any time a change upstream (IGNORE_LIGANDS, a filter fix, new hits)
# needs to be propagated through the full classification pipeline.
#
# USAGE:
#   python scripts/rerun_pipeline.py [--motif_dir DIR] [--include-modified]

import os
import sys
import glob
import json
import csv
import time
import argparse
from collections import Counter

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, SCRIPT_DIR)

import analyze_isosteres_simple
import analyze_plif
import sort_hits as sort_hits_module
import export_to_datawarrior


def _confirmed_key(fname):
    stem  = fname.replace(".pdb", "").replace(".cif", "")
    parts = stem.split("_")
    if len(parts) < 6:
        return None
    return (parts[1].upper(), parts[2].upper(), parts[3].upper(), parts[4].upper(), parts[5])


def _count_hits(d):
    if not os.path.exists(d):
        return 0
    return len(glob.glob(os.path.join(d, "hit_*.pdb"))) + len(glob.glob(os.path.join(d, "hit_*.cif")))


def _snapshot(motif_dir):
    hits_dir      = os.path.join(motif_dir, "hits")
    apo_dir       = os.path.join(hits_dir, "apo")
    holo_dir      = os.path.join(hits_dir, "holo")
    confirmed_dir = os.path.join(holo_dir, "plif_confirmed")

    confirmed_keys = {}
    if os.path.exists(confirmed_dir):
        for fpath in (glob.glob(os.path.join(confirmed_dir, "hit_*.pdb")) +
                      glob.glob(os.path.join(confirmed_dir, "hit_*.cif"))):
            key = _confirmed_key(os.path.basename(fpath))
            if key:
                confirmed_keys[key] = os.path.basename(fpath)

    plif_path   = os.path.join(motif_dir, "plif_results.json")
    plif_by_key = {}
    if os.path.exists(plif_path):
        with open(plif_path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            key = (r["ref"].upper(), r["hit"].upper(),
                   r.get("ref_chain", "").upper(), r.get("ref_lig", "").upper(),
                   str(r.get("ref_num", "")))
            plif_by_key[key] = r

    confirmed_details = {key: plif_by_key[key] for key in confirmed_keys if key in plif_by_key}

    csv_path = os.path.join(motif_dir, "isostere_full_list.csv")
    n_apo_csv = n_holo_csv = 0
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                state = row["State"].upper()
                if state == "APO":
                    n_apo_csv += 1
                elif state == "HOLO":
                    n_holo_csv += 1

    return {
        "apo_count":            _count_hits(apo_dir),
        "holo_count":           _count_hits(holo_dir),   # non-recursive: excludes plif_confirmed/ subfolder
        "confirmed_count":      len(confirmed_keys),
        "confirmed_keys":       set(confirmed_keys.keys()),
        "confirmed_details":    confirmed_details,
        "n_plif_results_total": len(plif_by_key),
        "isostere_csv_apo":     n_apo_csv,
        "isostere_csv_holo":    n_holo_csv,
    }


def _run_pipeline(motif_dir, include_modified):
    print("\n" + "=" * 78)
    print("STEP 1/4: analyze_isosteres_simple.py")
    print("=" * 78)
    analyze_isosteres_simple.analyze_isosteres_simple(motif_dir, include_modified=include_modified)

    print("\n" + "=" * 78)
    print("STEP 2/4: analyze_plif.py")
    print("=" * 78)
    analyze_plif.run_plif_analysis(motif_dir)

    print("\n" + "=" * 78)
    print("STEP 3/4: sort_hits.py")
    print("=" * 78)
    sort_hits_module.MOTIF_DIR = motif_dir
    sort_hits_module.HITS_DIR  = os.path.join(motif_dir, "hits")
    sort_hits_module.REFS_DIR  = os.path.join(motif_dir, "references")
    sort_hits_module.sort_hits()

    print("\n" + "=" * 78)
    print("STEP 4/4: export_to_datawarrior.py")
    print("=" * 78)
    export_to_datawarrior.export_for_datawarrior(
        results_path   = os.path.join(motif_dir, "plif_results.json"),
        smiles_path    = os.path.join(PROJECT_ROOT, "data", "Components-smiles-stereo-cactvs.smi"),
        output_path    = os.path.join(motif_dir, "isostere_datawarrior.txt"),
        confirmed_only = True,
        confirmed_dir  = os.path.join(motif_dir, "hits", "holo", "plif_confirmed"),
    )


def _avg_score(details):
    vals = [d.get("score", 0.0) for d in details.values()]
    return sum(vals) / len(vals) if vals else 0.0


def _build_report(before, after, elapsed):
    lines = []
    lines.append("\n" + "#" * 78)
    lines.append("# PIPELINE RUN SUMMARY")
    lines.append("#" * 78)
    lines.append(f"Elapsed time: {elapsed:.1f}s")

    def _row(label, b, a):
        return f"{label:<12} {b:>6}  ->  {a:>6}   (delta {a - b:+d})"

    lines.append("")
    lines.append(_row("Apo:",       before["apo_count"],       after["apo_count"]))
    lines.append(_row("Holo:",      before["holo_count"],      after["holo_count"]))
    lines.append(_row("Confirmed:", before["confirmed_count"], after["confirmed_count"]))

    lines.append("")
    lines.append(_row("CSV Apo:",  before["isostere_csv_apo"],  after["isostere_csv_apo"]))
    lines.append(_row("CSV Holo:", before["isostere_csv_holo"], after["isostere_csv_holo"]))

    lines.append("")
    lines.append(_row("PLIF entries (all, incl. below-threshold-excluded pairs no longer present):",
                       before["n_plif_results_total"], after["n_plif_results_total"]))

    added     = after["confirmed_keys"] - before["confirmed_keys"]
    removed   = before["confirmed_keys"] - after["confirmed_keys"]
    unchanged = after["confirmed_keys"] & before["confirmed_keys"]

    lines.append("")
    lines.append(f"Newly confirmed pairs:      {len(added)}")
    lines.append(f"No-longer-confirmed pairs:  {len(removed)}")
    lines.append(f"Unchanged confirmed pairs:  {len(unchanged)}")

    def _fmt_key(key, details_map):
        ref, hit, chain, lig, num = key
        d     = details_map.get(key, {})
        mimic = d.get("mimic", "?")
        score = d.get("score", 0.0)
        ec    = d.get("ec", "")
        return f"  {ref}/{hit}  chain={chain} lig={lig} num={num}  mimic={mimic:<6} score={score:.2f}  {ec}"

    if added:
        lines.append("\n--- NEW plif_confirmed pairs ---")
        for key in sorted(added):
            lines.append(_fmt_key(key, after["confirmed_details"]))

    if removed:
        lines.append("\n--- REMOVED plif_confirmed pairs ---")
        for key in sorted(removed):
            lines.append(_fmt_key(key, before["confirmed_details"]))

    before_mimics = Counter(d.get("mimic", "?") for d in before["confirmed_details"].values())
    after_mimics  = Counter(d.get("mimic", "?") for d in after["confirmed_details"].values())
    changed_codes = sorted(c for c in set(before_mimics) | set(after_mimics)
                           if before_mimics.get(c, 0) != after_mimics.get(c, 0))
    if changed_codes:
        lines.append("\n--- Confirmed mimic ligand counts that changed (before -> after) ---")
        for code in changed_codes:
            b, a = before_mimics.get(code, 0), after_mimics.get(code, 0)
            lines.append(f"  {code:<8} {b:>4}  ->  {a:>4}   (delta {a - b:+d})")

    lines.append("")
    lines.append(f"Distinct confirmed mimic codes: {len(before_mimics)}  ->  {len(after_mimics)}")
    lines.append(f"Average PLIF score (confirmed set): {_avg_score(before['confirmed_details']):.3f}  "
                 f"->  {_avg_score(after['confirmed_details']):.3f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Re-run analyze_isosteres_simple -> analyze_plif -> sort_hits -> "
                    "export_to_datawarrior and report what changed."
    )
    parser.add_argument("--motif_dir", default=os.path.join(PROJECT_ROOT, "results", "motif_analysis"))
    parser.add_argument("--include-modified", action="store_true",
                        help="Include modified residues as mimics (passed through to analyze_isosteres_simple)")
    args = parser.parse_args()

    motif_dir = args.motif_dir

    print("Taking snapshot of current state...")
    before = _snapshot(motif_dir)
    t0 = time.time()

    _run_pipeline(motif_dir, args.include_modified)

    elapsed = time.time() - t0
    print("\nTaking snapshot of new state...")
    after = _snapshot(motif_dir)

    report = _build_report(before, after, elapsed)
    print(report)

    log_dir = os.path.join(motif_dir, "pipeline_run_log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nFull report saved to {log_path}")


if __name__ == "__main__":
    main()
