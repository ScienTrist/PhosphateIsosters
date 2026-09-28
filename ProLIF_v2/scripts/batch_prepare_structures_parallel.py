"""
Parallelized version of batch_prepare_structures.py for full-scale manifests
(thousands of sites). PDB2PQR protein protonation is the dominant cost -- one
independent subprocess call per unique pdb_id, each reading/writing its own
files with no shared state -- so it's trivially parallelizable across a
thread pool instead of running serially. Threads (not processes) are enough
here since the actual work happens inside the pdb2pqr30 subprocess, which
releases the GIL while it runs.

Reuses the exact same per-id functions as the serial script (download,
protonate_protein.protonate, protonate_ligand.protonate_ligand,
fetch_ligand_smiles.fetch_smiles) -- not reimplemented -- so caching/
correctness behavior is identical to the 50-pair run; only wall-clock time
differs.

Runs in two phases rather than interleaving protein+ligand work per site:
  Phase 1: protonate every UNIQUE pdb_id's protein (dedup'd up front).
  Phase 2: protonate every site's ligand (2x the pdb_id count -- most PDB
           structures serve one candidate ligand, some contribute several
           distinct sites).
Two phases (instead of one pass doing both per site) means no site's ligand
protonation ever races another site sharing the same not-yet-protonated
protein -- Phase 1 fully completes first, so by Phase 2 every protein is
already on disk.

Does not run pocket extraction (extract_pockets.py) -- this script's scope is
download + protonation only, matching what was asked for; extraction is a
separate, fast, already-existing step to run afterward.
"""
import csv
import json
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import download_structures as dls  # noqa: E402
import protonate_protein as ppr  # noqa: E402
import protonate_ligand as plig  # noqa: E402
import fetch_ligand_smiles as fls  # noqa: E402

PROLIF_V2_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MANIFEST = os.path.join(PROLIF_V2_ROOT, "full_manifest.csv")
SMILES_CACHE_PATH = os.path.join(PROLIF_V2_ROOT, "ligand_smiles.json")

N_WORKERS_DEFAULT = min(os.cpu_count() or 4, 8)

_print_lock = threading.Lock()
_cache_lock = threading.Lock()
_progress = {"done": 0, "ok": 0, "fail": 0}

# Known failure signatures from this project's own log-forensics triage
# (see the session's pipeline-status writeup) -- anything that doesn't match
# one of these is tagged [NEW] instead of a category name, so a genuinely
# novel failure mode stands out live instead of blending into the ~200
# already-understood, already-excluded failures from the last full run.
KNOWN_FAILURE_PATTERNS = [
    ("missing_backbone_atoms", "Too few atoms present to reconstruct or cap residue"),
    ("non_integral_charge", "from integral"),
    ("nonetype_coords", "'NoneType' object has no attribute 'coords'"),
    ("debump_gap", "Unable to debump biomolecule"),
    ("rdkit_ligand_parse", "RDKit could not parse"),
    ("rdkit_ligand_parse", "Explicit valence"),
    ("ligand_not_found", "not found in chain"),
    ("no_raw_structure", "no .pdb or .cif for"),
]


def classify(msg):
    for tag, needle in KNOWN_FAILURE_PATTERNS:
        if needle in msg:
            return tag
    return "NEW"


def log(msg):
    with _print_lock:
        print(msg, flush=True)


sys.path.insert(0, os.path.join(os.path.dirname(PROLIF_V2_ROOT), "src"))
from utils import fmt_eta  # noqa: E402


def _report(item_id, ok, msg, total, t0, t_site, failures):
    tag = None if ok else classify(msg)
    with _print_lock:
        _progress["done"] += 1
        _progress["ok"] += ok
        _progress["fail"] += not ok
        done, n_ok, n_fail = _progress["done"], _progress["ok"], _progress["fail"]
        elapsed = time.time() - t0
        eta = elapsed / done * (total - done) if done else 0
        pct = 100 * done / total
        if ok:
            status = "OK"
        else:
            marker = "[NEW]" if tag == "NEW" else f"[{tag}]"
            status = f"FAILED {marker} {msg}"
            failures.append((item_id, tag, msg))
        print(f"  [{done}/{total} {pct:3.0f}%  ok={n_ok} fail={n_fail}] {item_id}: {status} "
              f"({time.time()-t_site:.1f}s this site, ETA {fmt_eta(eta)})", flush=True)


def print_failure_summary(phase_name, failures):
    if not failures:
        return
    by_tag = {}
    for item_id, tag, msg in failures:
        by_tag.setdefault(tag, []).append((item_id, msg))
    log(f"\n--- {phase_name} failure summary ({len(failures)} total) ---")
    # NEW first -- these are the ones that need eyes, not just a count
    ordered_tags = sorted(by_tag, key=lambda t: (t != "NEW", t))
    for tag in ordered_tags:
        items = by_tag[tag]
        flag = " <-- UNRECOGNIZED, LOOK AT THESE" if tag == "NEW" else ""
        log(f"  [{tag}] {len(items)}{flag}")
        if tag == "NEW":
            for item_id, msg in items:
                log(f"      {item_id}: {msg}")


def protonate_one_protein(pdb_id, total, t0, failures):
    t_site = time.time()
    try:
        dls.download(pdb_id)  # best-effort; protonate() falls back to a local .cif on failure
        ok, msg = ppr.protonate(pdb_id)
    except Exception as e:
        ok, msg = False, f"{type(e).__name__}: {e}"
    _report(pdb_id, ok, msg, total, t0, t_site, failures)
    return pdb_id, ok


def ensure_smiles(lig_code, cache):
    with _cache_lock:
        if lig_code in cache:
            return
    try:
        smiles = fls.fetch_smiles(lig_code)
    except Exception as e:
        log(f"      [smiles FAIL] {lig_code}: {e}")
        return
    with _cache_lock:
        cache[lig_code] = smiles


def protonate_one_ligand(site, cache, total, t0, failures):
    t_site = time.time()
    ensure_smiles(site["lig_resname"], cache)
    with _cache_lock:
        local_cache = dict(cache)  # protonate_ligand only reads; snapshot avoids lock during RDKit work
    try:
        outcome, msg = plig.protonate_ligand(site, local_cache)
        ok = outcome != "failed"
    except Exception as e:
        ok, msg = False, f"{type(e).__name__}: {e}"
    _report(site["site_id"], ok, msg, total, t0, t_site, failures)
    return site["site_id"], ok


def main():
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MANIFEST
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else N_WORKERS_DEFAULT

    sites = list(csv.DictReader(open(manifest_path)))
    unique_pdb_ids = sorted({s["pdb_id"] for s in sites})
    log(f"Manifest: {manifest_path}")
    log(f"{len(sites)} sites, {len(unique_pdb_ids)} unique PDB IDs, {n_workers} workers "
        f"({os.cpu_count()} CPUs available)")

    cache = json.load(open(SMILES_CACHE_PATH)) if os.path.exists(SMILES_CACHE_PATH) else {}

    log(f"\n=== Phase 1: protein protonation ({len(unique_pdb_ids)} unique structures) ===")
    _progress["done"] = 0
    _progress["ok"] = 0
    _progress["fail"] = 0
    t0 = time.time()
    phase1_failures = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(protonate_one_protein, pid, len(unique_pdb_ids), t0, phase1_failures)
                   for pid in unique_pdb_ids]
        for fut in as_completed(futures):
            fut.result()
    n_ok, n_fail = _progress["ok"], _progress["fail"]
    log(f"Phase 1 done in {fmt_eta(time.time()-t0)}: {n_ok}/{len(unique_pdb_ids)} proteins protonated, {n_fail} failed")
    print_failure_summary("Phase 1", phase1_failures)

    log(f"\n=== Phase 2: ligand protonation ({len(sites)} sites) ===")
    _progress["done"] = 0
    _progress["ok"] = 0
    _progress["fail"] = 0
    t0 = time.time()
    phase2_failures = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(protonate_one_ligand, s, cache, len(sites), t0, phase2_failures) for s in sites]
        for fut in as_completed(futures):
            fut.result()
    n_ok, n_fail = _progress["ok"], _progress["fail"]
    log(f"Phase 2 done in {fmt_eta(time.time()-t0)}: {n_ok}/{len(sites)} ligands protonated, {n_fail} failed")
    print_failure_summary("Phase 2", phase2_failures)

    with open(SMILES_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)

    log("\nAll done.")


if __name__ == "__main__":
    main()
