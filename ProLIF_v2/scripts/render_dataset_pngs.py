"""
Renders one highlighted PNG per row of a build_ifg_prolif_dataset.py output
CSV -- same red/purple convention as render_highlighted_hits.py (red = raw
ProLIF-matched atoms, purple = IFG-completion atoms) and the same filename
convention as the original highlighted_pngs_novdw run (<ref_pdb>_<ref_resnum>_
<hit_pdb>_<mimic>_<hit_resnum>.png).

Unlike render_highlighted_hits.py this does NOT recompute matched_hit_atom_
indices()/expand_to_functional_groups() (which need the cached ProLIF
Fingerprint + TM-align residue correspondence) -- it just reads the
red_atom_idxs/purple_atom_idxs columns build_ifg_prolif_dataset.py already
computed and rebuilds the same hit ligand mol (build_hit_ligand_mol, same
deterministic indexing) to draw against. Cheaper (no Fingerprint object to
load or ProLIF matching to redo) and guarantees the image matches the dataset
CSV exactly, not a second independent computation of the same thing.

Rows with no red atoms (no isosteric match -- ~half the dataset, see the
score=n/a/0.0 discussion) are skipped by default since there's nothing to
highlight; pass --include-empty to render them anyway (plain structure, no
highlight).

Parallelized the same way as build_ifg_prolif_dataset.py (ProcessPoolExecutor,
worker-local hit-mol cache persisting across the pool's lifetime) -- workers
here are lighter since no ProLIF Fingerprint pickle needs loading at all.

Usage: python render_dataset_pngs.py [dataset.csv] [--out-dir path]
                                      [--include-empty] [--workers N]
                                      [--error-log path]
"""
import argparse
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import csv  # noqa: E402
from export_prolif_datawarrior import PROLIF_V2_ROOT, RESULTS_DIR, _manifest_rows, build_hit_ligand_mol  # noqa: E402
from render_highlighted_hits import render  # noqa: E402
from utils import fmt_eta  # noqa: E402

WORKERS_DEFAULT = min(8, os.cpu_count() or 4)


def _parse_idxs(cell):
    return {int(x) for x in cell.split(",")} if cell else set()


_worker_manifest = None
_worker_hit_mol_cache = None


def _init_worker(manifest_path):
    global _worker_manifest, _worker_hit_mol_cache
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")  # see build_ifg_prolif_dataset._init_worker's comment
    _worker_manifest = _manifest_rows(manifest_path)
    _worker_hit_mol_cache = {}


def _render_row(row, out_dir):
    hit_sid = row["hit_site"]
    hit_row = _worker_manifest.get(hit_sid)
    if hit_row is None:
        return {"status": "fail", "row": row, "reason": f"{hit_sid} not in manifest"}

    red_idxs = _parse_idxs(row["red_atom_idxs"])
    purple_idxs = _parse_idxs(row["purple_atom_idxs"])

    try:
        if hit_sid not in _worker_hit_mol_cache:
            _worker_hit_mol_cache[hit_sid] = build_hit_ligand_mol(hit_row)
        lig_mol, mol_err = _worker_hit_mol_cache[hit_sid]
        if lig_mol is None:
            return {"status": "fail", "row": row, "reason": mol_err}

        ref_resnum = row["ref_site"].rsplit("_", 1)[-1]
        hit_resnum = hit_sid.rsplit("_", 1)[-1]
        fname = f"{row['ref_pdb']}_{ref_resnum}_{row['hit_pdb']}_{row['mimic']}_{hit_resnum}.png"
        out_path = os.path.join(out_dir, fname)
        render(lig_mol, red_idxs, purple_idxs, out_path)
    except Exception as e:
        return {"status": "fail", "row": row, "reason": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}

    return {"status": "ok", "fname": fname, "row": row}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?",
                         default=os.path.join(RESULTS_DIR, "ifg_prolif_dataset_full_manifest.csv"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--include-empty", action="store_true",
                         help="also render rows with no red atoms (plain structure, nothing highlighted)")
    parser.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    parser.add_argument("--error-log", default=None)
    parser.add_argument("--limit", type=int, default=0, help="max rows to render, 0 = all")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: {args.dataset} not found.")
        sys.exit(1)

    tag = os.path.splitext(os.path.basename(args.dataset))[0]
    prefix = "ifg_prolif_dataset"
    suffix = tag[len(prefix):] if tag.startswith(prefix) else f"_{tag}"
    manifest_name = f"{suffix[1:]}.csv" if suffix else "sample_manifest.csv"
    manifest_path = os.path.join(PROLIF_V2_ROOT, manifest_name)
    out_dir = args.out_dir or os.path.join(RESULTS_DIR, f"highlighted_pngs{suffix}")
    error_log_path = args.error_log or os.path.join(RESULTS_DIR, f"render_dataset_pngs_errors{suffix}.log")

    if not os.path.exists(manifest_path):
        print(f"Error: inferred manifest {manifest_path} not found -- pass --out-dir/--error-log "
              f"explicitly if the dataset filename doesn't follow ifg_prolif_dataset_<tag>.csv")
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.dataset, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not args.include_empty:
        rows = [r for r in rows if r["red_atom_idxs"]]
    if args.limit:
        rows = rows[:args.limit]

    total = len(rows)
    print(f"Rendering {total} PNGs from {args.dataset} ({args.workers} workers) -> {out_dir}")
    print(f"Errors -> {error_log_path}")

    n_ok = n_fail = 0
    t0 = time.time()
    errlog = open(error_log_path, "w", encoding="utf-8")
    errlog.write(f"# render_dataset_pngs.py -- dataset={args.dataset} workers={args.workers}\n")
    errlog.flush()

    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                  initargs=(manifest_path,)) as pool:
            futures = {pool.submit(_render_row, row, out_dir): row for row in rows}
            i = 0
            for future in as_completed(futures):
                row = futures[future]
                i += 1
                try:
                    result = future.result()
                except Exception as e:
                    result = {"status": "fail", "row": row, "reason": f"{type(e).__name__}: {e}",
                              "traceback": traceback.format_exc()}

                elapsed = time.time() - t0
                eta = elapsed / i * (total - i) if i else 0

                if result["status"] == "fail":
                    n_fail += 1
                    r = result["row"]
                    print(f"  [{i}/{total} {100*i/total:3.0f}%  ok={n_ok} fail={n_fail}] "
                          f"FAIL {r['ref_site']} vs {r['hit_site']}: {result['reason']} "
                          f"(ETA {fmt_eta(eta)})", flush=True)
                    errlog.write(f"[{i}/{total}] {r['ref_site']} vs {r['hit_site']}: {result['reason']}\n")
                    if result.get("traceback"):
                        errlog.write(result["traceback"])
                    errlog.write("-" * 70 + "\n")
                    errlog.flush()
                    continue

                n_ok += 1
                r = result["row"]
                print(f"  [{i}/{total} {100*i/total:3.0f}%  ok={n_ok} fail={n_fail}] "
                      f"{r['mimic']:8s} {r['ref_pdb']}->{r['hit_pdb']}  score={r['prolif_plif_score'] or 'n/a'}  "
                      f"-> {result['fname']} (ETA {fmt_eta(eta)})", flush=True)
    finally:
        errlog.write(f"\n# done: {n_ok} ok, {n_fail} failed out of {total}\n")
        errlog.close()

    print(f"\n{n_ok}/{total} PNGs written to {out_dir}, {n_fail} failed (see {error_log_path}).")


if __name__ == "__main__":
    main()
