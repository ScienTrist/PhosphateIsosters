"""
Draws new (ref, hit) candidate rows from the full isostere_full_list.csv pool
(the same pool select_sample.py drew its original 50 from, but excluding
Ref_IDs already tried there) and, for each, runs ONLY the reference-side
pipeline -- download, protonate protein+ligand, extract pocket -- far cheaper
than a full pair, before checking phosphate_ifp.best_phosphate_group() against
it. References whose best phosphate group makes < MIN_INTERACTIONS ProLIF-
detected interactions are dropped without ever touching the hit side.

Keeps going until N_NEEDED new qualifying references are found (or the
candidate pool / try cap is exhausted), then does the hit-side pipeline
(download, protonate, extract) for just those winners, and appends everything
to sample_manifest.csv alongside the existing 50 pairs.

Reuses download_structures.download, protonate_protein.protonate,
protonate_ligand.protonate_ligand, extract_pockets.extract_pocket, and
fetch_ligand_smiles.fetch_smiles as-is (all single-id/site functions, safe to
import since each module's expensive work only runs under its own
`if __name__ == "__main__"`), plus phosphate_ifp.best_phosphate_group for the
qualification check.
"""
import csv
import json
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")

PROLIF_V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PROLIF_V2_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "plip_isostere"))
sys.path.insert(0, os.path.join(PROLIF_V2_ROOT, "scripts"))

from common import FULL_LIST_CSV  # noqa: E402
from resolve_pairs import resolve_row  # noqa: E402

import download_structures as dls  # noqa: E402
import protonate_protein as ppr  # noqa: E402
import protonate_ligand as plig  # noqa: E402
import extract_pockets as epk  # noqa: E402
import fetch_ligand_smiles as fls  # noqa: E402
import phosphate_ifp as pif  # noqa: E402
import run_prolif as rp  # noqa: E402

MANIFEST_PATH = os.path.join(PROLIF_V2_ROOT, "sample_manifest.csv")
SMILES_CACHE_PATH = os.path.join(PROLIF_V2_ROOT, "ligand_smiles.json")
QUALIFYING_LOG = os.path.join(PROLIF_V2_ROOT, "results", "phosphate_expansion_log.csv")
os.makedirs(os.path.join(PROLIF_V2_ROOT, "results"), exist_ok=True)

MIN_INTERACTIONS = 3
N_NEEDED = 30          # new qualifying refs to look for (buffer above the ~27 shortfall)
MAX_TRIES = 220        # hard cap on candidate rows examined
SEED = 43              # different seed from select_sample.py's 42 -> fresh draw order


def make_site(role, resolved, paired_with):
    prefix = "ref" if role == "reference" else "hit"
    id_key, lig_key, num_key, chain_key = (
        ("Ref_ID", "Ref_Lig", "Ref_Num", "Ref_Chain") if role == "reference"
        else ("Hit_ID", "Hit_Ligand", "Hit_Num", "Hit_Chain")
    )
    return {
        "site_id": f"{prefix}_{resolved[id_key]}_{resolved[lig_key]}_{resolved[num_key]}",
        "role": role,
        "pdb_id": resolved[id_key],
        "chain": resolved[chain_key],
        "lig_resname": resolved[lig_key],
        "lig_resnum": str(resolved[num_key]),
        "paired_with": paired_with,
    }


def ensure_smiles(lig_code, cache):
    if lig_code in cache:
        return cache[lig_code]
    try:
        smiles = fls.fetch_smiles(lig_code)
        cache[lig_code] = smiles
        return smiles
    except Exception as e:
        print(f"    [smiles FAIL] {lig_code}: {e}")
        return None


def main():
    already_tried = {row["pdb_id"] for row in csv.DictReader(open(MANIFEST_PATH)) if row["role"] == "reference"}
    print(f"Excluding {len(already_tried)} Ref_IDs already present in sample_manifest.csv")

    rows = [r for r in csv.DictReader(open(FULL_LIST_CSV)) if r["Hit_Ligand"] not in ("None", "", None)]
    random.Random(SEED).shuffle(rows)
    rows = [r for r in rows if r["Ref_ID"] not in already_tried]
    print(f"{len(rows)} candidate rows available after exclusion")

    cache = json.load(open(SMILES_CACHE_PATH)) if os.path.exists(SMILES_CACHE_PATH) else {}

    import prolif as plf
    # rdkit preset covers all 118 elements vs mdanalysis's 54 -- see run_prolif.py's
    # Fingerprint construction for why (missing vdW radii otherwise crash the site).
    fp = plf.Fingerprint(rp.INTERACTIONS, parameters={"VdWContact": {"preset": "rdkit"}}, use_segid=False)

    qualifying = []   # list of dicts: resolved row + ref_site + hit_site + p_idx + n
    seen_refs = set()  # (pdb_id, lig, num) already attempted in this run
    n_tried = 0
    log_rows = []

    t0 = time.time()
    for row in rows:
        if len(qualifying) >= N_NEEDED or n_tried >= MAX_TRIES:
            break
        ref_key = (row["Ref_ID"], row["Ref_Lig"], row["Ref_Num"])
        if ref_key in seen_refs:
            continue
        seen_refs.add(ref_key)
        n_tried += 1

        resolved, err = resolve_row(row)
        if resolved is None:
            print(f"  [{n_tried}] [skip] {row['Ref_ID']}/{row['Hit_ID']} ({row['Hit_Ligand']}): {err}")
            continue

        ref_site = make_site("reference", resolved, resolved["Hit_ID"])

        try:
            if not dls.download(ref_site["pdb_id"]):
                raise RuntimeError("download failed")
            protein_ok, protein_msg = ppr.protonate(ref_site["pdb_id"])
            if not protein_ok:
                raise RuntimeError(f"protein protonation failed: {protein_msg}")
            ensure_smiles(ref_site["lig_resname"], cache)
            outcome, ligand_msg = plig.protonate_ligand(ref_site, cache)
            if outcome == "failed":
                raise RuntimeError(f"ligand protonation failed: {ligand_msg}")
            if not epk.extract_pocket(ref_site):
                raise RuntimeError("pocket extraction failed")

            p_idx, ifp, n, pf_err = pif.best_phosphate_group(ref_site, fp)
            if pf_err:
                raise RuntimeError(f"phosphate scoring failed: {pf_err}")
        except Exception as e:
            print(f"  [{n_tried}] [FAIL] {ref_site['site_id']}: {e}")
            log_rows.append({"site_id": ref_site["site_id"], "pdb_id": ref_site["pdb_id"],
                              "lig_resname": ref_site["lig_resname"], "n_interactions": "", "status": str(e)})
            continue

        log_rows.append({"site_id": ref_site["site_id"], "pdb_id": ref_site["pdb_id"],
                          "lig_resname": ref_site["lig_resname"], "n_interactions": n,
                          "status": "qualified" if n >= MIN_INTERACTIONS else "below_threshold"})

        if n >= MIN_INTERACTIONS:
            hit_site = make_site("hit", resolved, resolved["Ref_ID"])
            qualifying.append({"resolved": resolved, "ref_site": ref_site, "hit_site": hit_site,
                                "p_idx": p_idx, "n_interactions": n})
            print(f"  [{n_tried}] [OK] {ref_site['site_id']}: p_idx={p_idx} n_interactions={n} "
                  f"({len(qualifying)}/{N_NEEDED} found, {time.time()-t0:.0f}s elapsed)")
        else:
            print(f"  [{n_tried}] [low] {ref_site['site_id']}: n_interactions={n} (< {MIN_INTERACTIONS}, skipped)")

    with open(SMILES_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)

    with open(QUALIFYING_LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site_id", "pdb_id", "lig_resname", "n_interactions", "status"])
        w.writeheader()
        w.writerows(log_rows)

    print(f"\nTried {n_tried} candidates in {time.time()-t0:.0f}s, {len(qualifying)} new qualifying references found.")
    print(f"Wrote {QUALIFYING_LOG}")

    if not qualifying:
        print("Nothing to add to the manifest.")
        return

    # Hit-side prep for winners only, then append to sample_manifest.csv
    existing_rows = list(csv.DictReader(open(MANIFEST_PATH)))
    fieldnames = ["site_id", "role", "pdb_id", "chain", "lig_resname", "lig_resnum", "paired_with"]
    new_rows = []
    n_hit_ok = 0
    for q in qualifying:
        ref_site, hit_site = q["ref_site"], q["hit_site"]
        try:
            if not dls.download(hit_site["pdb_id"]):
                raise RuntimeError("hit download failed")
            protein_ok, protein_msg = ppr.protonate(hit_site["pdb_id"])
            if not protein_ok:
                raise RuntimeError(f"hit protein protonation failed: {protein_msg}")
            ensure_smiles(hit_site["lig_resname"], cache)
            outcome, ligand_msg = plig.protonate_ligand(hit_site, cache)
            if outcome == "failed":
                raise RuntimeError(f"hit ligand protonation failed: {ligand_msg}")
            if not epk.extract_pocket(hit_site):
                raise RuntimeError("hit pocket extraction failed")
        except Exception as e:
            print(f"  [hit FAIL] {hit_site['site_id']} (for {ref_site['site_id']}): {e}")
            continue
        new_rows.append(ref_site)
        new_rows.append(hit_site)
        n_hit_ok += 1

    with open(SMILES_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)

    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(existing_rows + new_rows)

    print(f"\n{n_hit_ok}/{len(qualifying)} new pairs fully prepped (ref+hit) and appended to {MANIFEST_PATH}")
    print(f"Manifest now has {len(existing_rows) + len(new_rows)} rows.")


if __name__ == "__main__":
    main()
