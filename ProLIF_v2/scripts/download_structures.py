"""
Step 2: downloads a fresh, unmodified .pdb file for every unique pdb_id in
sample_manifest.csv directly from RCSB (files.rcsb.org) -- deliberately not
reusing the copies already sitting in data/structures/{references,hits}/,
so this whole experiment has no dependency on anything the existing pipeline
(or PLIP) has touched.
"""
import csv
import os
import time
import urllib.request

PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(PROLIF_V2_ROOT, "sample_manifest.csv")
RAW_DIR = os.path.join(PROLIF_V2_ROOT, "data", "raw")

RCSB_URL = "https://files.rcsb.org/download/{}.pdb"
USER_AGENT = "Mozilla/5.0 (research script; phosphate-isostere-thesis)"


def download(pdb_id):
    out_path = os.path.join(RAW_DIR, f"{pdb_id}.pdb")
    if os.path.exists(out_path):
        print(f"  {pdb_id}: already downloaded, skipping")
        return True
    url = RCSB_URL.format(pdb_id)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except Exception as e:
        print(f"  {pdb_id}: FAILED ({e})")
        return False
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"  {pdb_id}: downloaded ({len(data)} bytes)")
    return True


def main():
    with open(MANIFEST_PATH) as f:
        pdb_ids = sorted(set(row["pdb_id"] for row in csv.DictReader(f)))

    print(f"Downloading {len(pdb_ids)} unique structures from RCSB...")
    ok, failed = 0, []
    for pdb_id in pdb_ids:
        if download(pdb_id):
            ok += 1
        else:
            failed.append(pdb_id)
        time.sleep(0.2)  # be polite to RCSB

    print(f"\nDone. {ok}/{len(pdb_ids)} downloaded.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
