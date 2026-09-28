"""
Builds a clustering-ready dataset of phosphate-isostere hits: for every ref/hit
pair in the ProLIF comparison CSV, identifies which atoms of the hit ("mimic")
ligand are actually responsible for replicating one of the reference phosphate
group's interactions (the "red" atoms -- same VdW-excluded ProLIF match
export_prolif_datawarrior.matched_hit_atom_indices already computes, matching
the highlighted_pngs_novdw convention), then uses the vendored IFG package
(ifg.py, Ertl 2017) to identify which whole functional group(s) those red atoms
belong to (the "purple" completion -- e.g. 2 of a sulfonate's 3 oxygens directly
interacting still pulls in the whole SO3 group).

Unlike render_highlighted_hits.py this does NOT render a PNG per pair -- only
SMILES/SMARTS text is written, which is enough to cluster on (by fingerprint or
by functional-group type) and, since it also stores the exact atom indices/
names, enough to regenerate a highlighted PNG for any single row on demand later
(reuse build_hit_ligand_mol(hit_row) + render_highlighted_hits.render(), or
explain_pair.py, with the stored red/purple atom indices).

Output columns per pair (one row):
  identifiers   ref_site, hit_site, ref_pdb, hit_pdb, ref_ligand, mimic
  scores        prolif_plif_score (recomputed, VdW-excluded), homebrew_plif_score,
                n_ca_mapped, same_phosphate_group, crystallographic_solvent,
                ref_n_bits_novdw (this pair's reference site's busiest-group
                non-VdW bit count -- see ref_busiest_group_novdw_bits())
  structure     full_ligand_smiles, total_heavy_atoms
  red atoms     red_atom_idxs, red_atom_names, red_heavy_atoms,
                red_fragment_smiles, red_fragment_smarts
                    -- the atoms actually behind a matched interaction
  IFG groups    ifg_group_atoms_smiles, ifg_group_type_smiles
                    -- whichever Ertl functional group(s) (ifg.py) at least one
                       red atom belongs to; "type" includes the attached
                       unmarked carbons (ifg.py's own convention), "atoms" is
                       the bare heteroatom cluster
  purple atoms  purple_atom_idxs, purple_atom_names, purple_heavy_atoms
                    -- IFG-completion atoms beyond the red set
  fractions     group_heavy_atoms, isosteric_fraction, group_fraction

Parallelized like run_prolif.py's own per-site fingerprinting phase: a
ProcessPoolExecutor whose workers each load the manifest + cached fingerprint
ONCE (in _init_worker, a module-level function -- required for Windows's
"spawn" start method, which pickles submitted work by qualified name, not by
value, so closures/nested functions can't be sent to workers) and keep their
own ref-phosphate-group/hit-mol/hit-IFG caches for the executor's whole
lifetime, so repeat ref_site or hit_site pairs landing on the same worker still
avoid recomputing phosphate_group_ifps()/build_hit_ligand_mol()/
identify_functional_groups(). Workers only ever return plain dicts/strings back
to the parent (never RDKit Mol or ProLIF Fingerprint objects), same discipline
run_prolif.py's _fingerprint_one_site follows.

--min-ref-bits filters out entire REFERENCE SITES (not just individual pairs)
whose busiest phosphate group has too few non-VdW interaction bits -- see
ref_busiest_group_novdw_bits()'s docstring for why this matters: a reference
with 0 such bits can never produce anything but score=n/a for every hit ever
paired against it, no matter how good the mimic actually is, and in the
full-manifest dataset just 41 such reference sites (of 388) were responsible
for 1268 of 2763 rows (46%) coming back n/a.

Usage: python build_ifg_prolif_dataset.py [manifest.csv] [--limit N]
                                           [--min-score S] [--min-ref-bits N]
                                           [--out path] [--error-log path]
                                           [--workers N]
"""
import argparse
import csv
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from export_prolif_datawarrior import (  # noqa: E402
    PROLIF_V2_ROOT, RESULTS_DIR, _manifest_rows, _comparison_rows,
    matched_hit_atom_indices, build_hit_ligand_mol, mol_to_clean_smiles,
    atom_names, CRYSTALLOGRAPHIC_SOLVENTS, _drop_excluded_interactions,
)
from ifg import identify_functional_groups  # noqa: E402
from utils import fmt_eta  # noqa: E402
import run_prolif as rp  # noqa: E402
import phosphate_ifp as pif  # noqa: E402

from rdkit import Chem  # noqa: E402

# Same cap-at-8 convention as run_prolif.py's FINGERPRINT_WORKERS (see that
# module's comment for the "why 8" precedent across this project's scripts).
WORKERS_DEFAULT = min(8, os.cpu_count() or 4)


def fragment_smiles_and_smarts(lig_mol, atom_idxs):
    """(smiles, smarts) of the atom-induced subgraph on atom_idxs, or ("", "")
    if atom_idxs is empty OR if the cut can't be sanitized. Both derived from
    the SAME extracted fragment mol (rp._extract_submol, valence-completed
    with implicit Hs -- see its docstring), so they describe the identical
    set of atoms/bonds.

    rp._extract_submol only catches Chem.KekulizeException, not the related
    AtomKekulizeException ("non-ring atom marked aromatic") -- cutting out
    just the raw red-matched atoms (as opposed to the FG-expanded set every
    other caller of _extract_submol uses) means the cut more often leaves 1-2
    atoms of an aromatic ring behind with a now-dangling aromatic flag, which
    trips this. Caught here rather than fixed in run_prolif.py (shared by
    several other scripts) -- a failed cut only means these two auxiliary
    columns come back empty for this one pair; the score/full-ligand-SMILES/
    IFG-group columns computed earlier don't depend on this and are still
    worth keeping instead of failing the whole row."""
    if not atom_idxs:
        return "", ""
    try:
        frag = rp._extract_submol(lig_mol, sorted(atom_idxs))
        smiles = mol_to_clean_smiles(frag)
        smarts = Chem.MolToSmarts(frag)
    except Exception:
        return "", ""
    return smiles, smarts


def ref_busiest_group_novdw_bits(ref_row, fp, ref_group_cache):
    """Non-VdW interaction bit count on this reference site's busiest
    phosphate group (by that same non-VdW bit count, not raw interaction
    count) -- a per-REFERENCE-SITE quality measure, independent of which hit
    it's paired against. This is deliberately a coarser, cheaper proxy than
    matched_hit_atom_indices's own per-pair group selection (which picks
    whichever group best overlaps THIS hit's bits) -- for a site-level
    richness filter that's applied identically to every hit sharing a
    reference, "does at least one of this reference's groups have real
    signal" is the right question, not "which group did this specific hit
    happen to line up with." Shares ref_group_cache's cache dict/keying with
    matched_hit_atom_indices (keyed by ref_row["site_id"]), so calling this
    first warms that cache for the per-pair matching that follows.

    Returns 0 if the reference has no phosphorus, no extracted pocket, or
    only VdWContact bits on every group -- all three cases where every pair
    against this reference would come back score=n/a either way."""
    ref_site_id = ref_row["site_id"]
    if ref_site_id not in ref_group_cache:
        ref_group_cache[ref_site_id] = pif.phosphate_group_ifps(ref_row, fp)
    groups, err = ref_group_cache[ref_site_id]
    if not groups:
        return 0
    best_n = 0
    for ifp in groups.values():
        bits = _drop_excluded_interactions(rp.flatten_canon_bits(rp.canonicalize_ifp_for_alignment(ifp)))
        best_n = max(best_n, len(bits))
    return best_n


def matched_ifg_groups(fgs, atom_idxs):
    """IFG groups (from ifg.identify_functional_groups) that share at least one
    atom with atom_idxs. Returns (group_atom_idxs: set[int], atoms_smiles: str,
    type_smiles: str) -- the latter two are ';'-joined if more than one group
    is touched (e.g. two separate carboxylates both catching a red atom)."""
    hit = [fg for fg in fgs if set(fg.atomIds) & atom_idxs]
    if not hit:
        return set(), "", ""
    group_idxs = set()
    for fg in hit:
        group_idxs.update(fg.atomIds)
    return group_idxs, "; ".join(fg.atoms for fg in hit), "; ".join(fg.type for fg in hit)


FIELDNAMES = [
    "ref_site", "hit_site", "ref_pdb", "hit_pdb", "ref_ligand", "mimic",
    "prolif_plif_score", "homebrew_plif_score", "n_ca_mapped", "same_phosphate_group",
    "crystallographic_solvent", "ref_n_bits_novdw",
    "full_ligand_smiles", "total_heavy_atoms",
    "red_atom_idxs", "red_atom_names", "red_heavy_atoms",
    "red_fragment_smiles", "red_fragment_smarts",
    "ifg_group_atoms_smiles", "ifg_group_type_smiles",
    "purple_atom_idxs", "purple_atom_names", "purple_heavy_atoms",
    "group_heavy_atoms", "isosteric_fraction", "group_fraction",
]


# ── Parallel per-pair worker ────────────────────────────────────────────────
# Module-level, not nested in main(): see docstring above (Windows spawn +
# picklability, same reasoning as run_prolif.py's own _init_worker).
_worker_manifest = None
_worker_fp = None
_worker_ifp_by_site = None
_worker_ref_group_cache = None
_worker_hit_mol_cache = None
_worker_hit_fgs_cache = None
_worker_ref_bits_by_site = None


def _init_worker(manifest_path, fp_pickle, ref_bits_by_site):
    """ProcessPoolExecutor initializer -- runs once per worker process. See
    run_prolif.py's _init_worker for why RDKit's native logger must be
    silenced here too: 8 worker processes all writing "Explicit valence.../
    Kekulize..." warnings to the same inherited stderr pipe concurrently has
    deadlocked this project's pool before.

    ref_bits_by_site is computed ONCE in main() (a small dict, one entry per
    unique reference site -- see main()'s pre-pass) and handed to every
    worker here rather than recomputed per-pair, so --min-ref-bits costs one
    extra ProLIF pass over unique ref sites total, not one per pair."""
    global _worker_manifest, _worker_fp, _worker_ifp_by_site
    global _worker_ref_group_cache, _worker_hit_mol_cache, _worker_hit_fgs_cache, _worker_ref_bits_by_site
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    import prolif as plf
    _worker_manifest = _manifest_rows(manifest_path)
    _worker_fp = plf.Fingerprint.from_pickle(fp_pickle)
    _worker_ifp_by_site = dict(zip(_worker_fp.site_ids, _worker_fp.ifp.values()))
    _worker_ref_group_cache = {}
    _worker_hit_mol_cache = {}
    _worker_hit_fgs_cache = {}
    _worker_ref_bits_by_site = ref_bits_by_site


def _process_pair(row):
    """Runs in a worker process. Returns a plain dict -- either
    {"status": "ok", "out_row": {...}, "ref_pdb":, "hit_pdb":, "mimic":,
    "score":, "red_heavy":, "group_heavy":, "total_heavy":} or
    {"status": "fail", "ref_sid":, "hit_sid":, "reason":, "traceback": str|None}
    -- never an RDKit Mol or ProLIF Fingerprint, so it's always cheaply
    picklable back to the parent."""
    ref_sid, hit_sid = row["ref_site"], row["hit_site"]
    ref_row = _worker_manifest.get(ref_sid)
    hit_row = _worker_manifest.get(hit_sid)
    if ref_row is None or hit_row is None:
        return {"status": "fail", "ref_sid": ref_sid, "hit_sid": hit_sid,
                "reason": "not in manifest", "traceback": None}

    hit_ifp = _worker_ifp_by_site.get(hit_sid)
    if hit_ifp is None:
        return {"status": "fail", "ref_sid": ref_sid, "hit_sid": hit_sid,
                "reason": f"{hit_sid} not in cached fingerprint", "traceback": None}

    try:
        atom_idxs, score, err = matched_hit_atom_indices(
            ref_row, hit_row, hit_ifp, _worker_fp, _worker_ref_group_cache)
        if err:
            return {"status": "fail", "ref_sid": ref_sid, "hit_sid": hit_sid,
                    "reason": err, "traceback": None}

        if hit_sid not in _worker_hit_mol_cache:
            _worker_hit_mol_cache[hit_sid] = build_hit_ligand_mol(hit_row)
        lig_mol, mol_err = _worker_hit_mol_cache[hit_sid]
        if lig_mol is None:
            return {"status": "fail", "ref_sid": ref_sid, "hit_sid": hit_sid,
                    "reason": mol_err, "traceback": None}

        if hit_sid not in _worker_hit_fgs_cache:
            _worker_hit_fgs_cache[hit_sid] = identify_functional_groups(lig_mol)
        fgs = _worker_hit_fgs_cache[hit_sid]

        group_idxs, ifg_atoms_smiles, ifg_type_smiles = matched_ifg_groups(fgs, atom_idxs)
        purple_idxs = group_idxs - atom_idxs

        full_smiles = mol_to_clean_smiles(lig_mol)
        total_heavy = sum(1 for a in lig_mol.GetAtoms() if a.GetAtomicNum() > 1)

        red_smiles, red_smarts = fragment_smiles_and_smarts(lig_mol, atom_idxs)
        red_names = atom_names(lig_mol, atom_idxs)
        red_heavy = sum(1 for idx in atom_idxs if lig_mol.GetAtomWithIdx(idx).GetAtomicNum() > 1)

        purple_names = atom_names(lig_mol, purple_idxs)
        purple_heavy = sum(1 for idx in purple_idxs if lig_mol.GetAtomWithIdx(idx).GetAtomicNum() > 1)

        group_heavy = red_heavy + purple_heavy
        isosteric_fraction = red_heavy / total_heavy if total_heavy else 0.0
        group_fraction = group_heavy / total_heavy if total_heavy else 0.0
    except Exception as e:
        return {"status": "fail", "ref_sid": ref_sid, "hit_sid": hit_sid,
                "reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}

    mimic = hit_row["lig_resname"]
    out_row = {
        "ref_site": ref_sid, "hit_site": hit_sid,
        "ref_pdb": ref_row["pdb_id"], "hit_pdb": hit_row["pdb_id"],
        "ref_ligand": ref_row["lig_resname"], "mimic": mimic,
        "prolif_plif_score": round(score, 3) if score is not None else "",
        "homebrew_plif_score": round(float(row["plif_score"]), 3) if row.get("plif_score") else "",
        "n_ca_mapped": row.get("n_ca_mapped", ""),
        "same_phosphate_group": row.get("same_phosphate_group", ""),
        "crystallographic_solvent": "Yes" if mimic in CRYSTALLOGRAPHIC_SOLVENTS else "No",
        "ref_n_bits_novdw": _worker_ref_bits_by_site.get(ref_sid, ""),
        "full_ligand_smiles": full_smiles,
        "total_heavy_atoms": total_heavy,
        "red_atom_idxs": ",".join(str(x) for x in sorted(atom_idxs)),
        "red_atom_names": ",".join(red_names),
        "red_heavy_atoms": red_heavy,
        "red_fragment_smiles": red_smiles,
        "red_fragment_smarts": red_smarts,
        "ifg_group_atoms_smiles": ifg_atoms_smiles,
        "ifg_group_type_smiles": ifg_type_smiles,
        "purple_atom_idxs": ",".join(str(x) for x in sorted(purple_idxs)),
        "purple_atom_names": ",".join(purple_names),
        "purple_heavy_atoms": purple_heavy,
        "group_heavy_atoms": group_heavy,
        "isosteric_fraction": round(isosteric_fraction, 3),
        "group_fraction": round(group_fraction, 3),
    }
    return {"status": "ok", "out_row": out_row, "ref_pdb": ref_row["pdb_id"],
            "hit_pdb": hit_row["pdb_id"], "mimic": mimic, "score": score,
            "red_heavy": red_heavy, "group_heavy": group_heavy, "total_heavy": total_heavy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="full_manifest.csv")
    parser.add_argument("--limit", type=int, default=0,
                         help="max pairs to process, sorted by ProLIF PLIF score descending (0 = all)")
    parser.add_argument("--min-score", type=float, default=0.0,
                         help="skip pairs with prolif_plif_score below this")
    parser.add_argument("--min-ref-bits", type=int, default=0,
                         help="skip entire reference sites whose busiest phosphate group has "
                              "<= this many non-VdW interaction bits (see module docstring)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--error-log", default=None)
    parser.add_argument("--workers", type=int, default=WORKERS_DEFAULT,
                         help=f"parallel worker processes (default {WORKERS_DEFAULT})")
    args = parser.parse_args()

    manifest_path = os.path.join(PROLIF_V2_ROOT, args.manifest)
    manifest_tag = os.path.splitext(os.path.basename(manifest_path))[0]
    run_suffix = "" if manifest_tag == "sample_manifest" else f"_{manifest_tag}"
    comparison_csv = os.path.join(RESULTS_DIR, f"plif_vs_tanimoto_comparison{run_suffix}.csv")
    fp_pickle = os.path.join(RESULTS_DIR, f"prolif_fingerprint{run_suffix}.pkl")
    out_path = args.out or os.path.join(RESULTS_DIR, f"ifg_prolif_dataset{run_suffix}.csv")
    error_log_path = args.error_log or os.path.join(RESULTS_DIR, f"ifg_prolif_dataset_errors{run_suffix}.log")

    for p in (manifest_path, comparison_csv, fp_pickle):
        if not os.path.exists(p):
            print(f"Error: {p} not found.")
            sys.exit(1)

    rows = _comparison_rows(comparison_csv)
    rows = [r for r in rows if r["prolif_plif_score"] not in ("", None)
            and float(r["prolif_plif_score"]) >= args.min_score]
    rows.sort(key=lambda r: float(r["prolif_plif_score"]), reverse=True)

    manifest = _manifest_rows(manifest_path)
    unique_ref_sites = sorted({r["ref_site"] for r in rows})

    # Reuse a cached ref_site -> non-VdW-bit-count CSV if one covers every
    # unique ref site this run needs -- this pre-pass takes ~5 min sequential
    # (one ProLIF fp.generate() per unique reference site) and its result
    # depends only on --min-score (which ref sites are even in play), not on
    # --min-ref-bits itself, so re-running it from scratch every time the
    # threshold is tweaked is pure waste. Falls back to computing (and then
    # writing this same cache) if missing or incomplete.
    ref_bits_cache_path = os.path.join(RESULTS_DIR, f"ref_novdw_bits_by_site{run_suffix}.csv")
    ref_bits_by_site = {}
    if os.path.exists(ref_bits_cache_path):
        with open(ref_bits_cache_path, newline="", encoding="utf-8") as f:
            cached = {r["ref_site"]: int(r["ref_n_bits_novdw"]) for r in csv.DictReader(f)}
        if set(unique_ref_sites) <= set(cached):
            ref_bits_by_site = {sid: cached[sid] for sid in unique_ref_sites}
            print(f"Reusing cached non-VdW ref-bit counts from {ref_bits_cache_path} "
                  f"({len(ref_bits_by_site)} ref sites, pre-pass skipped)")
        else:
            missing = len(set(unique_ref_sites) - set(cached))
            print(f"Cache at {ref_bits_cache_path} is missing {missing}/{len(unique_ref_sites)} "
                  f"ref sites needed -- recomputing")

    if not ref_bits_by_site:
        print(f"Loading cached ProLIF fingerprint from {fp_pickle} for the --min-ref-bits pre-pass ...")
        import prolif as plf
        fp = plf.Fingerprint.from_pickle(fp_pickle)
        ref_group_cache = {}
        for ref_sid in unique_ref_sites:
            ref_row = manifest.get(ref_sid)
            ref_bits_by_site[ref_sid] = ref_busiest_group_novdw_bits(ref_row, fp, ref_group_cache) if ref_row else 0
        with open(ref_bits_cache_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ref_site", "ref_n_bits_novdw"])
            for sid in unique_ref_sites:
                w.writerow([sid, ref_bits_by_site[sid]])
        print(f"Cached non-VdW ref-bit counts -> {ref_bits_cache_path}")

    n_ref_pass = sum(1 for n in ref_bits_by_site.values() if n > args.min_ref_bits)
    print(f"  {n_ref_pass}/{len(unique_ref_sites)} reference sites have > {args.min_ref_bits} non-VdW bits "
          f"on their busiest phosphate group")

    if args.min_ref_bits:
        before = len(rows)
        rows = [r for r in rows if ref_bits_by_site.get(r["ref_site"], 0) > args.min_ref_bits]
        print(f"  --min-ref-bits {args.min_ref_bits}: kept {len(rows)}/{before} pairs "
              f"({len(unique_ref_sites) - n_ref_pass} reference sites dropped)")

    if args.limit:
        rows = rows[:args.limit]

    out_rows = []
    n_ok = n_fail = 0
    t0 = time.time()
    total = len(rows)
    print(f"Building IFG/ProLIF dataset for {total} pairs (limit={args.limit or 'none'}, "
          f"min_score={args.min_score}, min_ref_bits={args.min_ref_bits}, {args.workers} workers) ...")
    print(f"Errors -> {error_log_path}")

    errlog = open(error_log_path, "w", encoding="utf-8")
    errlog.write(f"# build_ifg_prolif_dataset.py -- manifest={args.manifest} "
                 f"min_score={args.min_score} min_ref_bits={args.min_ref_bits} "
                 f"limit={args.limit or 'none'} workers={args.workers}\n")
    errlog.flush()

    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                  initargs=(manifest_path, fp_pickle, ref_bits_by_site)) as pool:
            futures = {pool.submit(_process_pair, row): row for row in rows}
            i = 0
            for future in as_completed(futures):
                row = futures[future]
                ref_sid, hit_sid = row["ref_site"], row["hit_site"]
                i += 1
                try:
                    result = future.result()
                except Exception as e:
                    result = {"status": "fail", "ref_sid": ref_sid, "hit_sid": hit_sid,
                              "reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}

                elapsed = time.time() - t0
                eta = elapsed / i * (total - i) if i else 0

                if result["status"] == "fail":
                    n_fail += 1
                    print(f"  [{i}/{total} {100*i/total:3.0f}%  ok={n_ok} fail={n_fail}] "
                          f"FAIL {result['ref_sid']} vs {result['hit_sid']}: {result['reason']} "
                          f"(ETA {fmt_eta(eta)})", flush=True)
                    errlog.write(f"[{i}/{total}] {result['ref_sid']} vs {result['hit_sid']}: {result['reason']}\n")
                    if result["traceback"]:
                        errlog.write(result["traceback"])
                    errlog.write("-" * 70 + "\n")
                    errlog.flush()
                    continue

                n_ok += 1
                score_str = f"{result['score']:.3f}" if result["score"] is not None else "n/a"
                print(f"  [{i}/{total} {100*i/total:3.0f}%  ok={n_ok} fail={n_fail}] "
                      f"{result['ref_pdb']}/{result['hit_pdb']} ({result['mimic']}): score={score_str} "
                      f"red={result['red_heavy']}/{result['total_heavy']} "
                      f"group={result['group_heavy']}/{result['total_heavy']} heavy atoms "
                      f"(ETA {fmt_eta(eta)})", flush=True)
                out_rows.append(result["out_row"])
    finally:
        errlog.write(f"\n# done: {n_ok} ok, {n_fail} failed out of {total}\n")
        errlog.close()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\n{n_ok}/{total} pairs processed, {n_fail} failed (see {error_log_path}).")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
