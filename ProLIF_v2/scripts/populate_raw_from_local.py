"""
Seeds ProLIF_v2/data/raw/ by copying already-downloaded raw PDB files from
the main pipeline's local cache (data/structures/references/, data/structures/
hits/) instead of re-fetching from RCSB -- at full-manifest scale (thousands
of structures) redownloading everything download_structures.py-style would be
a large amount of redundant network traffic for files already sitting on disk.

Only copies files actually referenced by the given manifest (default:
full_manifest.csv) -- NOT the entire data/structures/ pool (8,339 references +
31,023 hits on disk), which would pull in thousands of structures never used
by any resolved candidate pair.

Copies each pdb_id from the category its role implies (reference -> data/
structures/references/, hit -> data/structures/hits/), matching exactly how
resolve_pairs.py's find_pdb_path() looks them up -- a pdb_id's category is
fixed by which set (seeds vs homologs) it was downloaded into originally, so
there's no ambiguity about which source directory to read from.

After this, download_structures.py's own download() (called from
batch_prepare_structures.py) sees the file already present at data/raw/ and
skips the RCSB fetch entirely -- no changes needed to the protonation scripts.
"""
import csv
import os
import shutil
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROLIF_V2_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PROLIF_V2_ROOT)

RAW_DIR = os.path.join(PROLIF_V2_ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

SOURCE_DIRS = {
    "reference": os.path.join(PROJECT_ROOT, "data", "structures", "references"),
    "hit": os.path.join(PROJECT_ROOT, "data", "structures", "hits"),
}

DEFAULT_MANIFEST = os.path.join(PROLIF_V2_ROOT, "full_manifest.csv")


def log(msg):
    print(msg, flush=True)


def main():
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MANIFEST
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    # dedupe by (pdb_id, role) -- one physical raw file per pdb_id, but keep
    # role since reference/hit pdb_ids are sourced from different directories
    needed = {}
    for r in rows:
        needed[(r["pdb_id"], r["role"])] = True

    log(f"Manifest: {len(rows)} rows -> {len(needed)} unique (pdb_id, role) raw files needed")

    n_copied = n_cached = n_missing = n_cif = 0
    t0 = time.time()
    for i, (pdb_id, role) in enumerate(needed, 1):
        # Some structures only exist as .cif locally (a handful aren't even
        # available as legacy .pdb from RCSB at all, e.g. 9QW4 404s on the
        # .pdb URL) -- protonate_protein.py handles a .cif input itself (via
        # a gemmi conversion, since PDB2PQR's own CIF parser crashes on real
        # files), so preserve whichever extension the source actually has
        # rather than only ever looking for .pdb.
        if os.path.exists(os.path.join(RAW_DIR, f"{pdb_id}.pdb")) or os.path.exists(os.path.join(RAW_DIR, f"{pdb_id}.cif")):
            n_cached += 1
            continue

        src_pdb = os.path.join(SOURCE_DIRS[role], f"{pdb_id.upper()}.pdb")
        src_cif = os.path.join(SOURCE_DIRS[role], f"{pdb_id.upper()}.cif")
        if os.path.exists(src_pdb):
            src, ext = src_pdb, "pdb"
        elif os.path.exists(src_cif):
            src, ext = src_cif, "cif"
            n_cif += 1
        else:
            log(f"  [{i}/{len(needed)}] [MISSING] {pdb_id} ({role}): not found at {src_pdb} or {src_cif}")
            n_missing += 1
            continue

        dest = os.path.join(RAW_DIR, f"{pdb_id}.{ext}")
        shutil.copyfile(src, dest)
        n_copied += 1
        if n_copied % 200 == 0:
            elapsed = time.time() - t0
            log(f"  [{i}/{len(needed)}] copied {n_copied} so far ({elapsed:.0f}s elapsed)")

    log(f"\nDone in {time.time()-t0:.0f}s: {n_copied} copied ({n_cif} as .cif, no .pdb available), "
        f"{n_cached} already present, {n_missing} missing from local source.")
    if n_missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
