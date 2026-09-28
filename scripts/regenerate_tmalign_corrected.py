"""
Generates a NEW tmalign_results_{run_id}_corrected.json using the corrected
claiming logic now in main.py's run_concurrent_workflow (quality-based
hit->seed assignment + starved-seed rescue), WITHOUT touching the original
tmalign_results_{run_id}.json at all.

Does this by copying the original file to the new path first, then pointing
TMAlignHandler.set_checkpoint() at the COPY. submit_alignment() already skips
any (hit, seed) pair already present in whatever checkpoint it's given, so
the ~18,861 pairs that are still valid under the corrected logic are reused
for free, and only the genuinely new ~9,836 pairs actually get computed --
same resume-safe mechanism main.py itself relies on, just aimed at a
different output file.

See main.py's run_concurrent_workflow for the full reasoning behind the
claiming logic itself (best e-value/identity wins a contested hit, starved
seeds get their own best candidate forced in).
"""
import json
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tmalign_handler import TMAlignHandler  # noqa: E402

RUN_ID = "3D_no_2.7_0.25_Xray_16_0.5_0.8_7.5_0.95_0.8"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ENRICHMENT_FILE = os.path.join(RESULTS_DIR, f"homology_enrichment_{RUN_ID}.json")
ORIGINAL_TM_FILE = os.path.join(RESULTS_DIR, f"tmalign_results_{RUN_ID}.json")
NEW_TM_FILE = os.path.join(RESULTS_DIR, f"tmalign_results_{RUN_ID}_corrected.json")


def build_target_pairs(enrichment_results):
    """Same logic as main.py's run_concurrent_workflow: single assignment per
    hit (best e-value, ties broken by highest identity), plus a rescue pair
    for any seed that would otherwise end up with zero coverage."""
    hit_to_seed = {}
    hit_quality = {}
    for seed, data in enrichment_results["cluster_hits"].items():
        seed_pdb = seed[:4].upper()
        for hit in data["hits"]:
            hit_pdb = hit["pdb_id"].upper()
            quality = (hit.get("evalue", float("inf")), -hit.get("identity", 0.0))
            if hit_pdb not in hit_to_seed or quality < hit_quality[hit_pdb]:
                hit_to_seed[hit_pdb] = seed_pdb
                hit_quality[hit_pdb] = quality

    covered_seeds = set(hit_to_seed.values())
    rescue_by_hit = {}
    n_rescued = 0
    for seed, data in enrichment_results["cluster_hits"].items():
        seed_pdb = seed[:4].upper()
        if seed_pdb in covered_seeds or not data["hits"]:
            continue
        best_hit = min(
            data["hits"],
            key=lambda h: (h.get("evalue", float("inf")), -h.get("identity", 0.0)),
        )
        rescue_by_hit.setdefault(best_hit["pdb_id"].upper(), []).append(seed_pdb)
        n_rescued += 1

    pairs = set((seed, hit) for hit, seed in hit_to_seed.items())
    for hit, seeds in rescue_by_hit.items():
        for seed in seeds:
            pairs.add((seed, hit))

    print(f"  single-assignment pairs: {len(hit_to_seed)}")
    print(f"  seeds rescued: {n_rescued}")
    print(f"  total target pairs: {len(pairs)}")
    return pairs


def main():
    if not os.path.exists(ORIGINAL_TM_FILE):
        print(f"Error: {ORIGINAL_TM_FILE} not found.")
        sys.exit(1)

    print(f"Copying {os.path.basename(ORIGINAL_TM_FILE)} -> {os.path.basename(NEW_TM_FILE)} "
          f"(original left untouched from here on)...")
    shutil.copyfile(ORIGINAL_TM_FILE, NEW_TM_FILE)

    with open(ENRICHMENT_FILE) as f:
        enrichment_results = json.load(f)

    print("Building corrected target pair set...")
    target_pairs = build_target_pairs(enrichment_results)

    handler = TMAlignHandler(PROJECT_ROOT, fast_mode=False)
    handler.set_checkpoint(NEW_TM_FILE)

    print(f"\nSubmitting {len(target_pairs)} target pairs "
          f"(already-satisfied ones are skipped automatically)...")
    with handler:
        for seed_pdb, hit_pdb in target_pairs:
            handler.submit_alignment(hit_pdb, seed_pdb)
        handler.collect_results(wait_for_all=True)

    print(f"\nDone. Wrote {NEW_TM_FILE}")
    print(f"Original {ORIGINAL_TM_FILE} was never opened for writing.")


if __name__ == "__main__":
    main()
