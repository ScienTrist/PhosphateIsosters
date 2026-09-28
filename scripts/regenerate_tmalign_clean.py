"""
Builds a fully fresh tmalign_results_{RUN_ID}_clean.json from scratch.

Unlike regenerate_tmalign_corrected.py, this does NOT copy or reuse any
existing tmalign_results file. Every one of the current target pairs
(single-assignment + starved-seed rescue, identical logic to main.py's
run_concurrent_workflow) is recomputed from zero into a brand-new
checkpoint file, so the result cannot inherit whatever caused the old
_corrected.json to end up missing ~9,700 target pairs while still
carrying ~17,900 pairs left over from a since-abandoned pairing scheme.
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tmalign_handler import TMAlignHandler  # noqa: E402

RUN_ID = "3D_no_2.7_0.25_Xray_16_0.5_0.8_7.5_0.95_0.8"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ENRICHMENT_FILE = os.path.join(RESULTS_DIR, f"homology_enrichment_{RUN_ID}.json")
CLEAN_TM_FILE = os.path.join(RESULTS_DIR, f"tmalign_results_{RUN_ID}_clean.json")


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
    if os.path.exists(CLEAN_TM_FILE):
        print(f"Error: {CLEAN_TM_FILE} already exists. Refusing to overwrite -- "
              f"delete it first if you really want a fresh run.")
        sys.exit(1)

    with open(ENRICHMENT_FILE) as f:
        enrichment_results = json.load(f)

    print("Building target pair set...")
    target_pairs = build_target_pairs(enrichment_results)

    # fast_mode=False to match main.py's TM_FAST_MODE = False setting
    handler = TMAlignHandler(PROJECT_ROOT, fast_mode=False)
    # Checkpoint path does not exist yet -> set_checkpoint() finds nothing to
    # load, guaranteeing zero reuse of any prior result.
    handler.set_checkpoint(CLEAN_TM_FILE)
    assert not handler._results, "Checkpoint unexpectedly loaded existing data -- aborting."

    print(f"\nSubmitting all {len(target_pairs)} target pairs for FRESH computation...")
    with handler:
        for seed_pdb, hit_pdb in target_pairs:
            handler.submit_alignment(hit_pdb, seed_pdb)
        handler.collect_results(wait_for_all=True)

    print(f"\nDone. Wrote {CLEAN_TM_FILE}")

    # Final verification pass
    with open(CLEAN_TM_FILE) as f:
        final = json.load(f)
    result_pairs = set()
    for seed, hits in final.items():
        for h in hits:
            result_pairs.add((seed.upper(), h["hit"].upper()))
    missing = target_pairs - result_pairs
    extra = result_pairs - target_pairs
    print(f"Verification: {len(result_pairs)} pairs in output file.")
    print(f"  Missing vs target: {len(missing)}")
    print(f"  Extra vs target:   {len(extra)}")
    if missing:
        print(f"  Sample missing (likely genuine TM-align failures): {list(missing)[:10]}")


if __name__ == "__main__":
    main()
