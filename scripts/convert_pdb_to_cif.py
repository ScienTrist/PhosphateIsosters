"""
Converts all extracted hit_*.pdb and ref_*.pdb files to mmCIF format in-place.

Why: PDB format truncates residue names to 3 characters; mmCIF has no such limit.
New extractions already write .cif directly. This script migrates existing files.

Note: residue names already truncated in the .pdb files remain truncated after
conversion — the truncation is baked into the content. Only a full re-extraction
would restore names like A1EIC. For analysis purposes this is fine because the
CA backbone check already filters those modified amino acids out.
"""
import os
import sys
import glob
import gemmi
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def convert_file(pdb_path):
    cif_path = pdb_path[:-4] + ".cif"
    try:
        st = gemmi.read_structure(pdb_path)
        st.make_mmcif_document().write_file(cif_path)
        os.remove(pdb_path)
        return pdb_path, "ok"
    except Exception as e:
        if os.path.exists(cif_path):
            os.remove(cif_path)
        return pdb_path, f"error: {e}"


def main():
    motif_dir = os.path.join(PROJECT_ROOT, "results", "motif_analysis")
    patterns = [
        os.path.join(motif_dir, "references", "ref_*.pdb"),
        os.path.join(motif_dir, "hits", "**", "hit_*.pdb"),
    ]

    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat, recursive=True))

    total = len(all_files)
    print(f"Found {total} PDB files to convert.")
    if total == 0:
        print("Nothing to do.")
        return

    ok = errors = 0
    workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(convert_file, p): p for p in all_files}
        for i, future in enumerate(as_completed(futures), 1):
            _, status = future.result()
            if status == "ok":
                ok += 1
            else:
                errors += 1
                print(f"  ERROR: {futures[future]} — {status}")

            if i % 1000 == 0 or i == total:
                print(f"  [{i}/{total}] converted={ok}  errors={errors}", flush=True)

    print(f"\nDone. {ok} files converted, {errors} errors.")


if __name__ == "__main__":
    main()
