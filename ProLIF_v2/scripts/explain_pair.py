"""
Explains ONE already-fingerprinted ref/hit site pair: for both the homebrew
scorer (analyze_plif.py) and ProLIF (run_prolif.py), lists every reference
interaction and whether it was matched somewhere in the hit.

Neither run_prolif.py nor compare_with_homebrew_plif.py keeps this breakdown
around -- both reduce it to a single float per pair (plif_score /
prolif_plif_score) and move on, since keeping every pair's full breakdown in
memory for the whole full-manifest run isn't worth it. This script recomputes
it for exactly one pair, on demand, reusing the cached
prolif_fingerprint_full_manifest.pkl (avoids re-running ProLIF's interaction
detection on 5,572 sites just to explain one pair) plus a fresh homebrew call
(cheap -- two small pocket files, no heavy fingerprinting).

Usage: python explain_pair.py <ref_site_id> <hit_site_id> [manifest.csv]
Prints one JSON object to stdout:
{
  "homebrew": {"score": float, "interactions": [{"residue": str, "type": str, "matched": bool}, ...]},
  "prolif":   {"score": float|null, "interactions": [{"residue": str, "type": str, "matched": bool}, ...]}
}
"""
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROLIF_V2_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT   = os.path.dirname(PROLIF_V2_ROOT)
RESULTS_DIR    = os.path.join(PROLIF_V2_ROOT, "results")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "plip_isostere"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, SCRIPT_DIR)

import gemmi  # noqa: E402
from analyze_plif import (  # noqa: E402
    get_interactions_per_phosphate, get_interactions,
    calculate_spatial_similarity, get_metal_positions,
)
import run_prolif as rp  # noqa: E402
import phosphate_ifp as pif  # noqa: E402


def _manifest_rows(manifest_name):
    path = os.path.join(PROLIF_V2_ROOT, manifest_name)
    with open(path) as f:
        return {r["site_id"]: r for r in csv.DictReader(f)}


_comparison_cache = None


def _ground_truth_score(ref_site, hit_site, column):
    """Looks up an already-published score for this exact pair from
    plif_vs_tanimoto_comparison_full_manifest.csv (column is "plif_score" for
    homebrew or "prolif_plif_score" for ProLIF), purely as a sanity check on the
    live recomputation below -- both should agree, since they're supposed to be
    the same calculation. None if the pair isn't in that file or the file is missing."""
    global _comparison_cache
    if _comparison_cache is None:
        _comparison_cache = {}
        path = os.path.join(RESULTS_DIR, "plif_vs_tanimoto_comparison_full_manifest.csv")
        if os.path.exists(path):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    _comparison_cache[(row["ref_site"], row["hit_site"])] = row
    row = _comparison_cache.get((ref_site, hit_site))
    if row is None or row[column] == "":
        return None
    return float(row[column])


def _explain_homebrew(ref_row, hit_row):
    ref_path = os.path.join(rp.POCKETS_DIR, f"{ref_row['site_id']}.pdb")
    hit_path = os.path.join(rp.POCKETS_DIR, f"{hit_row['site_id']}.pdb")
    st_ref = gemmi.read_structure(ref_path)
    st_hit = gemmi.read_structure(hit_path)

    # Same per-P-atom grouping definition ProLIF's phosphate_ifp.py uses (P atom
    # plus O/N within 2.1 A) -- the group COUNT (not the interactions) doubles as
    # a cheap, deterministic stand-in for "does this reference have more than one
    # phosphate group", reused below by _explain_prolif instead of a second,
    # separate ProLIF-side recomputation (see that function's docstring for why
    # a live ProLIF recomputation is avoided).
    ref_inters_per_p = get_interactions_per_phosphate(st_ref, ref_row["lig_resname"], ref_row["lig_resnum"])
    n_phosphate_groups = len(ref_inters_per_p)
    if ref_inters_per_p:
        _, ref_inters = max(ref_inters_per_p.items(), key=lambda kv: len(kv[1]))
    else:
        ref_inters = []

    hit_inters = get_interactions(st_hit, hit_row["lig_resname"], hit_row["lig_resnum"])
    hit_metals = get_metal_positions(st_hit)
    mimic_positions = []
    for chain in st_hit[0]:
        for res in chain:
            if res.name == hit_row["lig_resname"] and str(res.seqid.num) == str(hit_row["lig_resnum"]):
                mimic_positions = [(a.pos.x, a.pos.y, a.pos.z) for a in res]

    # calculate_spatial_similarity matches purely by raw 3D distance (< 2.0 A,
    # no residue-identity check at all) -- it therefore REQUIRES ref and hit
    # positions to already be in the same frame. ProLIF_v2/data/pockets/*.pdb
    # (unlike results/motif_analysis/, which compare_with_homebrew_plif.py reads
    # and where hits were pre-superimposed at extraction time) is written in
    # each site's own NATIVE crystallographic frame -- confirmed directly: TYD3002's
    # and THM301's ligand atoms sit ~50 A apart in raw coordinates, in unrelated
    # frames. Without this transform every homebrew match check silently fails
    # regardless of true similarity, which is exactly what an earlier version of
    # this function did (0/3 matched on a pair ProLIF scored 8/10 on the same
    # residues) -- same convention as run_prolif.py's build_residue_correspondence
    # and pymol_load_pair_simple.py: apply the pipeline's saved (t, u) TM-align
    # transform to the hit's coordinates, reference stays fixed.
    transformation = rp.get_transformation(ref_row["pdb_id"], hit_row["pdb_id"])
    transform_error = None
    if transformation is None:
        transform_error = f"no TM-align transform on file for ({ref_row['pdb_id']}, {hit_row['pdb_id']})"
        # Can't spatially compare two structures with no known relative frame --
        # drop the hit side entirely rather than leave it in its native
        # (meaningless-for-comparison) frame, where a coincidental sub-2.0-A
        # overlap could produce a spurious "matched" result.
        hit_inters, hit_metals, mimic_positions = [], [], []
    else:
        t, u = transformation
        hit_inters = [(rp.apply_transform(t, u, pos), itype, res, num, chain)
                      for (pos, itype, res, num, chain) in hit_inters]
        hit_metals = [(rp.apply_transform(t, u, pos), res, num, chain)
                      for (pos, res, num, chain) in hit_metals]
        mimic_positions = [rp.apply_transform(t, u, pos) for pos in mimic_positions]

    score, matched_pairs = calculate_spatial_similarity(ref_inters, hit_inters, hit_metals, mimic_positions)

    rows = []
    for (pos, itype, res, num, chain) in ref_inters:
        key = (itype, res, num, chain)
        rows.append({"residue": f"{res}{num}.{chain}", "type": itype, "matched": key in matched_pairs})
    rows.sort(key=lambda r: (r["residue"], r["type"]))
    result = {"score": score, "interactions": rows, "n_phosphate_groups": n_phosphate_groups}
    if transform_error:
        # Not "error" -- score/interactions are still meaningful (score is
        # correctly 0.0 / all-unmatched, same as calculate_spatial_similarity's
        # own behavior on an empty hit_inters), just flagged so the caller
        # doesn't mistake "no transform" for "genuinely zero overlap".
        result["group_caveat"] = transform_error

    truth = _ground_truth_score(ref_row["site_id"], hit_row["site_id"], "plif_score")
    if truth is not None and score is not None and abs(truth - score) > 0.01:
        result["score_check"] = (f"WARNING: recomputed score {score:.3f} does not match published "
                                  f"plif_score {truth:.3f} for this pair -- treat this breakdown with caution")
    return result


def _explain_prolif(ref_row, hit_row, fp, ifp_by_site):
    # The hit is always whole-ligand (never phosphorus-containing by construction),
    # so its cached site-level ifp is exactly what run_prolif.py itself uses --
    # no ambiguity, unlike the reference side below.
    hit_ifp = ifp_by_site.get(hit_row["site_id"])
    if hit_ifp is None:
        return {"score": None, "interactions": [], "error": f"{hit_row['site_id']} not in cached fingerprint"}

    mapping = rp.build_residue_correspondence(ref_row, hit_row, cutoff=rp.CA_MATCH_CUTOFF)
    if mapping is None:
        return {"score": None, "interactions": [], "error": f"no TM-align transform for ({ref_row['pdb_id']}, {hit_row['pdb_id']})"}
    canon_hit_ifp = rp.canonicalize_ifp_for_alignment(hit_ifp, protein_mapping=mapping)
    hit_bits = rp.flatten_canon_bits(canon_hit_ifp)

    # Reference groups are recomputed live (not read from the cached site-level
    # ifp): when a reference ligand has more than one phosphate group (e.g. TYD's
    # alpha/beta diphosphate), run_prolif.py's own per-pair scoring does NOT use
    # the cached "busiest group overall" ifp (ifp_by_site) -- it picks whichever
    # group has the most matched interactions against THIS specific hit
    # (ref_groups_by_site in run_prolif.py's main(), never written to disk).
    # Reproduced here the same way, then cross-checked against the already
    # -published prolif_plif_score for this exact pair (see "score_check" below)
    # to catch any divergence rather than assume it.
    groups, err = pif.phosphate_group_ifps(ref_row, fp)
    if groups is None:
        return {"score": None, "interactions": [], "error": f"phosphate_group_ifps failed: {err}"}
    if not groups:
        return {"score": None, "interactions": [], "error": "reference ligand has no phosphorus atom"}

    if len(groups) == 1:
        ref_ifp = next(iter(groups.values()))
    else:
        best_ifp, best_matched = None, -1
        for cand_ifp in groups.values():
            cand_bits = rp.flatten_canon_bits(rp.canonicalize_ifp_for_alignment(cand_ifp))
            matched = len(cand_bits & hit_bits)
            if matched > best_matched:
                best_ifp, best_matched = cand_ifp, matched
        ref_ifp = best_ifp

    canon_ref_ifp = rp.canonicalize_ifp_for_alignment(ref_ifp)
    ref_bits = rp.flatten_canon_bits(canon_ref_ifp)

    rows = []
    for (residue_str, itype) in ref_bits:
        rows.append({"residue": residue_str, "type": itype, "matched": (residue_str, itype) in hit_bits})
    rows.sort(key=lambda r: (r["residue"], r["type"]))

    score = len(ref_bits & hit_bits) / len(ref_bits) if ref_bits else None
    result = {"score": score, "interactions": rows}

    truth = _ground_truth_score(ref_row["site_id"], hit_row["site_id"], "prolif_plif_score")
    if truth is not None and score is not None and abs(truth - score) > 0.01:
        result["score_check"] = (f"WARNING: recomputed score {score:.3f} does not match published "
                                  f"prolif_plif_score {truth:.3f} for this pair -- treat this breakdown with caution")
    return result


def explain(ref_site, hit_site, manifest_name="full_manifest.csv"):
    manifest = _manifest_rows(manifest_name)
    if ref_site not in manifest or hit_site not in manifest:
        return {"error": f"{ref_site} or {hit_site} not found in {manifest_name}"}
    ref_row, hit_row = manifest[ref_site], manifest[hit_site]
    homebrew = _explain_homebrew(ref_row, hit_row)
    homebrew.pop("n_phosphate_groups", None)

    fp_pickle = os.path.join(RESULTS_DIR, "prolif_fingerprint_full_manifest.pkl")
    if not os.path.exists(fp_pickle):
        prolif = {"score": None, "interactions": [], "error": f"missing {fp_pickle}"}
    else:
        import prolif as plf
        fp = plf.Fingerprint.from_pickle(fp_pickle)
        ifp_by_site = dict(zip(fp.site_ids, fp.ifp.values()))
        prolif = _explain_prolif(ref_row, hit_row, fp, ifp_by_site)

    return {"homebrew": homebrew, "prolif": prolif}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: explain_pair.py <ref_site_id> <hit_site_id> [manifest.csv]"}))
        sys.exit(1)
    ref_site, hit_site = sys.argv[1], sys.argv[2]
    manifest_name = sys.argv[3] if len(sys.argv) > 3 else "full_manifest.csv"
    print(json.dumps(explain(ref_site, hit_site, manifest_name)))
