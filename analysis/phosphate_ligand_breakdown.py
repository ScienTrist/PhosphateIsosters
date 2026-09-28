"""
Counts, per phosphate-like ligand code, how many PDB structures contain it
-- computed AFTER Step 1's quality filters (resolution/R-free/method) and
ligand filter (3D phosphate-geometry match, no_po4 mode), i.e. over the same
21,592-structure set in results/ligand_search_<run_id>.json.

Neither that file nor any other cache in this pipeline records per-structure
ligand IDENTITY (query_rcsb's RCSB search only returns matching PDB codes,
not which candidate ligand code each one actually matched -- see
src/rcsb_search.py), so this script re-derives it in two steps:

  1. Re-run get_phosphate_ligands_3d() against the local Chemical Component
     Dictionary (data/components.cif) to regenerate the same phosphate-like
     ligand code vocabulary Step 1 used (purely local, ~15s).
  2. Batch-query RCSB's Data API (rcsbapi.data.DataQuery, same pattern
     src/uniprot_mapping.py already uses elsewhere in this codebase) for the
     real nonpolymer ligand composition of all 21,592 structures, then
     intersect each structure's composition against the candidate
     vocabulary to recover which specific phosphate ligand(s) it has.

A structure with more than one distinct matching ligand is counted once per
ligand (so per-ligand counts can sum to more than the structure count).

Usage: python analysis/phosphate_ligand_breakdown.py
"""
import collections
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from ligand_detection_3D import get_phosphate_ligands_3d  # noqa: E402

RUN_ID = "3D_no_2.7_0.25_Xray_16_0.5_0.8_7.5_0.95_0.8"
LIGAND_SEARCH_JSON = os.path.join(PROJECT_ROOT, "results", f"ligand_search_{RUN_ID}.json")
COMPONENTS_CIF = os.path.join(PROJECT_ROOT, "data", "components.cif")
QUERY_MODE = "no_po4"  # must match run_params.QUERY_MODE in the ligand_search JSON

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
PHOSPHATE_CODES_CACHE = os.path.join(CACHE_DIR, f"phosphate_ligand_codes_{QUERY_MODE}.json")
PDB_TO_LIGANDS_CACHE = os.path.join(CACHE_DIR, f"pdb_to_ligands_{RUN_ID}.json")
OUT_CSV = os.path.join(PROJECT_ROOT, "results", f"phosphate_ligand_counts_{RUN_ID}.csv")

BATCH_SIZE = 500


def get_phosphate_codes():
    if os.path.exists(PHOSPHATE_CODES_CACHE):
        with open(PHOSPHATE_CODES_CACHE) as f:
            return set(json.load(f))
    print(f"Re-deriving phosphate ligand vocabulary from {COMPONENTS_CIF} ...")
    codes = get_phosphate_ligands_3d(COMPONENTS_CIF, mode=QUERY_MODE)
    with open(PHOSPHATE_CODES_CACHE, "w") as f:
        json.dump(sorted(codes), f)
    return codes


def get_pdb_ligand_composition(pdb_ids):
    if os.path.exists(PDB_TO_LIGANDS_CACHE):
        with open(PDB_TO_LIGANDS_CACHE) as f:
            cached = json.load(f)
        if set(pdb_ids) <= set(cached):
            return cached
    from rcsbapi.data import DataQuery

    print(f"Fetching ligand composition for {len(pdb_ids)} structures from RCSB Data API ...")
    pdb_to_ligs = {}
    t0 = time.time()
    for i in range(0, len(pdb_ids), BATCH_SIZE):
        batch = pdb_ids[i:i + BATCH_SIZE]
        try:
            dq = DataQuery(input_type="entry", input_ids=batch, return_data_list=[
                "rcsb_id",
                "nonpolymer_entities.rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
            ])
            res = dq.exec()
            entries = (res.get("data", {}).get("entries", []) if res else [])
            for entry in entries:
                if not entry:
                    continue
                pid = entry.get("rcsb_id")
                nps = entry.get("nonpolymer_entities") or []
                comp_ids = set()
                for np_entity in nps:
                    cid = (np_entity.get("rcsb_nonpolymer_entity_container_identifiers") or {}).get("nonpolymer_comp_id")
                    if cid:
                        comp_ids.add(cid)
                pdb_to_ligs[pid] = sorted(comp_ids)
        except Exception as e:
            print(f"  batch {i}-{i+BATCH_SIZE} FAILED: {type(e).__name__}: {e}")
        done = min(i + BATCH_SIZE, len(pdb_ids))
        print(f"  [{done}/{len(pdb_ids)}] elapsed={time.time()-t0:.0f}s")

    with open(PDB_TO_LIGANDS_CACHE, "w") as f:
        json.dump(pdb_to_ligs, f)
    return pdb_to_ligs


def main():
    with open(LIGAND_SEARCH_JSON) as f:
        data = json.load(f)
    pdb_ids = data["high_quality_phosphate_pdb_codes"]
    print(f"{len(pdb_ids)} quality+ligand-filtered structures (run_params: {data['run_params']})")

    phosphate_codes = get_phosphate_codes()
    print(f"{len(phosphate_codes)} candidate phosphate-like ligand codes")

    pdb_to_ligs = get_pdb_ligand_composition(pdb_ids)

    ligand_counts = collections.Counter()
    n_multi = n_zero = 0
    for pdb, ligs in pdb_to_ligs.items():
        present = set(ligs) & phosphate_codes
        if not present:
            n_zero += 1
            continue
        if len(present) > 1:
            n_multi += 1
        for lig in present:
            ligand_counts[lig] += 1

    print(f"\nStructures with >=1 matching phosphate ligand: {len(pdb_to_ligs) - n_zero}")
    print(f"Structures with >1 distinct phosphate ligand: {n_multi}")
    print(f"Structures with no cross-matching ligand (CCD/RCSB drift): {n_zero}")
    print(f"Distinct phosphate ligand codes observed: {len(ligand_counts)}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write("ligand_code,structure_count\n")
        for lig, count in ligand_counts.most_common():
            f.write(f"{lig},{count}\n")
    print(f"\nWrote {OUT_CSV}")

    print("\nTop 30:")
    for lig, count in ligand_counts.most_common(30):
        print(f"  {lig:8} {count}")


if __name__ == "__main__":
    main()
