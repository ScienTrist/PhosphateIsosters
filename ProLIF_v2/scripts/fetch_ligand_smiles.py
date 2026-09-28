"""
Fetches the reference SMILES for every distinct ligand code in the manifest
from RCSB's Chemical Component Dictionary (data.rcsb.org) -- this is the
"template" ProLIF's docs recommend combining with PDB coordinates via
rdkit.Chem.AllChem.AssignBondOrdersFromTemplate to get correct bond orders/
formal charges before adding explicit hydrogens (see protonate_ligand.py).

Caches results to ligand_smiles.json so re-runs don't re-hit the network.
"""
import csv
import json
import os
import time
import urllib.request

PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(PROLIF_V2_ROOT, "sample_manifest.csv")
CACHE_PATH = os.path.join(PROLIF_V2_ROOT, "ligand_smiles.json")

CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{}"
USER_AGENT = "Mozilla/5.0 (research script; phosphate-isostere-thesis)"


def fetch_smiles(lig_code):
    req = urllib.request.Request(CHEMCOMP_URL.format(lig_code), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    desc = data.get("rcsb_chem_comp_descriptor", {})
    smiles = desc.get("SMILES_stereo") or desc.get("SMILES")
    if not smiles:
        raise ValueError("no SMILES in rcsb_chem_comp_descriptor")
    return smiles


def main():
    with open(MANIFEST_PATH) as f:
        lig_codes = sorted(set(row["lig_resname"] for row in csv.DictReader(f)))

    cache = {}
    if os.path.exists(CACHE_PATH):
        cache = json.load(open(CACHE_PATH))

    print(f"Fetching SMILES for {len(lig_codes)} distinct ligand codes...")
    failed = []
    for code in lig_codes:
        if code in cache:
            print(f"  {code}: cached")
            continue
        try:
            smiles = fetch_smiles(code)
            cache[code] = smiles
            print(f"  {code}: {smiles}")
        except Exception as e:
            print(f"  {code}: FAILED ({e})")
            failed.append(code)
        time.sleep(0.2)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)

    print(f"\nDone. {len(cache)}/{len(lig_codes)} resolved. Wrote {CACHE_PATH}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
