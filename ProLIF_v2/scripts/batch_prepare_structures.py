"""
Consolidated batch downloader + protonator for the final sample_manifest.csv
(50 ref/hit pairs), with live progress reporting. Replaces manually chaining
download_structures.py -> protonate_protein.py -> protonate_ligand.py one at
a time -- reuses those modules' per-id functions as-is (not reimplemented),
just drives them site by site with flushed progress output and cleans up the
raw (unprotonated) download afterward, since nothing downstream of this step
(extract_pockets.py) ever reads data/raw/ again once both protein and ligand
are protonated.

No RCSB batch-download shell script exists anywhere in this repo (checked
via glob for **/*.sh) -- download_structures.py's single-file urllib fetch
already serves as this project's batch downloader, called once per PDB ID.

Reference and hit sites are processed as two explicit, separately-reported
groups (progress lines are labeled "(reference)"/"(hit)" and the final
summary counts them separately) -- the manifest's site_id prefix already
encodes role unambiguously (ref_.../hit_...), and both roles land in the
same flat data/protein_protonated/, data/ligand_protonated/ directories
extract_pockets.py already expects, so this doesn't restructure storage,
just makes the role split visible in the run.
"""
import csv
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import download_structures as dls  # noqa: E402
import protonate_protein as ppr  # noqa: E402
import protonate_ligand as plig  # noqa: E402
import fetch_ligand_smiles as fls  # noqa: E402

PROLIF_V2_ROOT = os.path.dirname(SCRIPT_DIR)
MANIFEST_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROLIF_V2_ROOT, "sample_manifest.csv")
SMILES_CACHE_PATH = os.path.join(PROLIF_V2_ROOT, "ligand_smiles.json")
RAW_DIR = dls.RAW_DIR


def log(msg):
    print(msg, flush=True)


sys.path.insert(0, os.path.join(os.path.dirname(PROLIF_V2_ROOT), "src"))
from utils import fmt_eta  # noqa: E402


def prepare_site(site, cache, protein_done_pdbids):
    """Downloads (if needed), protonates protein (once per pdb_id) and ligand
    for one manifest site. Returns (ok, msg): msg is the specific failure
    reason (protein- or ligand-side) when ok is False, for live display."""
    dls.download(site["pdb_id"])  # best-effort; protonate() falls back to a local .cif on failure

    if site["pdb_id"] not in protein_done_pdbids:
        success, msg = ppr.protonate(site["pdb_id"])
        if not success:
            return False, msg
        protein_done_pdbids.add(site["pdb_id"])

    if site["lig_resname"] not in cache:
        try:
            cache[site["lig_resname"]] = fls.fetch_smiles(site["lig_resname"])
        except Exception as e:
            log(f"      [smiles FAIL] {site['lig_resname']}: {e}")

    outcome, msg = plig.protonate_ligand(site, cache)
    return outcome != "failed", msg


def cleanup_raw(sites_by_pdbid, manifest_pdb_ids):
    """Deletes data/raw/{pdb_id}.pdb once nothing needs it any more: either
    every site for that pdb_id in the current manifest has a protonated
    protein + ligand on disk, or the pdb_id isn't in the manifest at all any
    more (a leftover from an earlier, discarded candidate draw)."""
    if not os.path.isdir(RAW_DIR):
        return 0
    n_deleted, n_kept = 0, 0
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith(".pdb"):
            continue
        pdb_id = fname[:-4]
        raw_path = os.path.join(RAW_DIR, fname)

        if pdb_id not in manifest_pdb_ids:
            os.remove(raw_path)
            n_deleted += 1
            continue

        protein_path = os.path.join(ppr.PROTEIN_DIR, f"{pdb_id}_protein.pdb")
        sites = sites_by_pdbid.get(pdb_id, [])
        ligs_ok = all(
            os.path.exists(os.path.join(plig.OUT_DIR, f"{s['site_id']}_ligand.pdb"))
            for s in sites
        )
        if os.path.exists(protein_path) and ligs_ok:
            os.remove(raw_path)
            n_deleted += 1
        else:
            n_kept += 1
    log(f"Deleted {n_deleted} raw PDB(s); kept {n_kept} whose protonation isn't fully complete yet.")
    return n_deleted


def main():
    sites = list(csv.DictReader(open(MANIFEST_PATH)))
    refs = [s for s in sites if s["role"] == "reference"]
    hits = [s for s in sites if s["role"] == "hit"]
    log(f"Manifest: {len(refs)} reference sites, {len(hits)} hit sites ({len(sites)} total)")

    cache = json.load(open(SMILES_CACHE_PATH)) if os.path.exists(SMILES_CACHE_PATH) else {}

    sites_by_pdbid = {}
    for s in sites:
        sites_by_pdbid.setdefault(s["pdb_id"], []).append(s)

    n_total = len(sites)
    n_done = 0
    n_ok = {"reference": 0, "hit": 0}
    n_fail = {"reference": 0, "hit": 0}
    protein_done_pdbids = set()
    t0 = time.time()

    for role, group in (("reference", refs), ("hit", hits)):
        log(f"\n=== {role} sites ({len(group)}) ===")
        for site in group:
            t_site = time.time()
            n_done += 1
            pct = 100 * n_done / n_total
            elapsed = time.time() - t0
            eta = elapsed / n_done * (n_total - n_done) if n_done else 0
            log(f"[{n_done}/{n_total} {pct:3.0f}%] ({role}) {site['site_id']} ...")

            ok, msg = prepare_site(site, cache, protein_done_pdbids)
            n_ok[role] += ok
            n_fail[role] += not ok
            status = "OK" if ok else f"FAILED ({msg})"
            log(f"    {status}  ({time.time()-t_site:.1f}s this site, ETA {fmt_eta(eta)} remaining)")

    with open(SMILES_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)

    log(f"\nreference: {n_ok['reference']}/{len(refs)} prepared, hit: {n_ok['hit']}/{len(hits)} prepared "
        f"({time.time()-t0:.0f}s total)")

    log("\nCleaning up now-unneeded raw (unprotonated) downloads...")
    manifest_pdb_ids = set(sites_by_pdbid.keys())
    cleanup_raw(sites_by_pdbid, manifest_pdb_ids)

    if n_fail["reference"] or n_fail["hit"]:
        log(f"\n{n_fail['reference'] + n_fail['hit']} site(s) failed -- see FAILED lines above.")
        sys.exit(1)
    log("\nAll sites prepared successfully.")


if __name__ == "__main__":
    main()
