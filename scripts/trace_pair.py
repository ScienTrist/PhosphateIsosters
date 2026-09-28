# Trace a single ref/hit pair through the PLIF pipeline and report exactly
# which stage it passes or fails at. Use this instead of re-reading
# analyze_isosteres_simple.py / analyze_plif.py / sort_hits.py by hand whenever
# a pair's apo/holo/plif_confirmed placement looks wrong.
#
# USAGE:
#   python scripts/trace_pair.py <REF_ID> <HIT_ID> [--chain X] [--lig XXX] [--num N]
#
# If a ref/hit pair has multiple reference sites (different chain/lig/num),
# and --chain/--lig/--num aren't given, all matching sites are listed so you
# can re-run with the specific one you want.

import os
import sys
import glob
import csv
import json
import argparse

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, SCRIPT_DIR)

import gemmi
from constants import IGNORE_LIGANDS, STANDARD_AA, DISTANCE_CUTOFFS
import analyze_plif as ap
import sort_hits as sh


def _find_hit_files(hits_dir, ref_id, hit_id, chain, lig, num):
    all_files = (glob.glob(os.path.join(hits_dir, "**", "hit_*.pdb"), recursive=True) +
                 glob.glob(os.path.join(hits_dir, "**", "hit_*.cif"), recursive=True))
    matches = []
    for fpath in all_files:
        stem  = os.path.basename(fpath).replace(".pdb", "").replace(".cif", "")
        parts = stem.split("_")
        if len(parts) < 6:
            continue
        p_ref, p_hit, p_chain, p_lig, p_num = parts[1], parts[2], parts[3], parts[4], parts[5]
        if p_ref.upper() != ref_id.upper() or p_hit.upper() != hit_id.upper():
            continue
        if chain and p_chain.upper() != chain.upper():
            continue
        if lig and p_lig.upper() != lig.upper():
            continue
        if num and str(p_num) != str(num):
            continue
        matches.append(fpath)
    return sorted(set(matches))


def _mimic_filter_reasons(res, hit_p_positions=None):
    """Reproduce analyze_plif._process_ref_site's candidate-mimic filter,
    returning a list of reasons the residue would be EXCLUDED (empty = passes)."""
    has_p = any(a.element.name == "P" for a in res)
    is_aa = gemmi.find_tabulated_residue(res.name).is_amino_acid()
    reasons = []
    if has_p:
        reasons.append("contains phosphorus (excluded from mimic scoring — isostere search)")
    if hit_p_positions:
        link_cutoff = DISTANCE_CUTOFFS["LINKED_PHOSPHATE"]
        if any(a.pos.dist(p) < link_cutoff for a in res for p in hit_p_positions):
            reasons.append(f"covalently/closely linked (<{link_cutoff} A) to a phosphorus atom "
                            "in a different residue (e.g. ARA-TT7 style split ligand)")
    if is_aa:
        reasons.append("is an amino acid (canonical or modified)")
    if res.name in IGNORE_LIGANDS:
        reasons.append("in IGNORE_LIGANDS")
    if res.name in STANDARD_AA:
        reasons.append("in STANDARD_AA")
    return reasons


def _check_recorded(ref_id, hit_id, chain, ref_lig, ref_num, motif_dir):
    csv_path = os.path.join(motif_dir, "isostere_full_list.csv")
    csv_state = None
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["Ref_ID"].upper() == ref_id.upper() and row["Hit_ID"].upper() == hit_id.upper():
                    csv_state = row
        if csv_state:
            print(f"    isostere_full_list.csv: State={csv_state['State']}  "
                  f"Hit_Ligand={csv_state.get('Hit_Ligand')}  Dist={csv_state.get('Distance')}  "
                  f"Pocket_RMSD={csv_state.get('Pocket_RMSD')}")
            print("      (note: this CSV is keyed by ref/hit pair only, not the full site — "
                  "if this pair has multiple reference sites they share one row)")
        else:
            print("    isostere_full_list.csv: no row for this ref/hit pair.")
    else:
        print("    isostere_full_list.csv: not found.")

    json_path = os.path.join(motif_dir, "plif_results.json")
    plif_rows = []
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        plif_rows = [
            r for r in data
            if r["ref"].upper() == ref_id.upper() and r["hit"].upper() == hit_id.upper()
            and r.get("ref_chain", chain).upper() == chain.upper()
            and r.get("ref_lig", ref_lig).upper() == ref_lig.upper()
            and str(r.get("ref_num", ref_num)) == str(ref_num)
        ]
        if plif_rows:
            for r in plif_rows:
                print(f"    plif_results.json: mimic={r['mimic']}  score={r['score']:.3f}  "
                      f"rmsd={r['rmsd']:.2f}  P{r.get('ref_p_idx', '?')}")
        else:
            print("    plif_results.json: no matching entry — analyze_plif.py did not confirm this "
                  "site (or hasn't been re-run since the underlying data/filters changed).")
    else:
        print("    plif_results.json: not found.")

    verify_dir = os.path.join(motif_dir, "plif_verification")
    stale = sorted(glob.glob(os.path.join(verify_dir, f"verify_{ref_id}_{hit_id}_*.txt")))
    if stale:
        print(f"    plif_verification/: {len(stale)} file(s) for this ref/hit pair:")
        for s in stale:
            note = "" if plif_rows else "  <-- no corresponding plif_results.json entry (stale/left over from a previous run)"
            print(f"      {os.path.basename(s)}{note}")

    is_holo = csv_state is not None and csv_state["State"].upper() == "HOLO"
    confirmed_pairs, use_site_keys = sh.load_confirmed_pairs(motif_dir)
    if use_site_keys:
        match_key = (ref_id.upper(), hit_id.upper(), chain.upper(), ref_lig.upper(), str(ref_num))
    else:
        match_key = (ref_id.upper(), hit_id.upper())
    is_confirmed = is_holo and match_key in confirmed_pairs
    verdict = "plif_confirmed" if is_confirmed else ("holo (unconfirmed)" if is_holo else "apo")
    print(f"\n  [Verdict] sort_hits.py would currently classify this site as: {verdict}")


def _trace_one(hit_path, hits_dir, refs_dir, motif_dir):
    fname = os.path.basename(hit_path)
    stem  = fname.replace(".pdb", "").replace(".cif", "")
    parts = stem.split("_")
    ref_id, hit_id, chain, ref_lig, ref_num = parts[1], parts[2], parts[3], parts[4], parts[5]

    print("\n" + "=" * 78)
    print(f"  SITE   ref={ref_id}  hit={hit_id}  chain={chain}  lig={ref_lig}  num={ref_num}")
    print("=" * 78)
    print(f"  Current file location: hits/{os.path.relpath(os.path.dirname(hit_path), hits_dir)}/")

    ref_path = ap._find_ref_path(refs_dir, ref_id, chain, ref_lig, ref_num)
    if not ref_path:
        print(f"  STOP — reference file not found (tried ref_{ref_id}_{chain}_{ref_lig}_{ref_num}.[cif|pdb])")
        return
    print(f"  Ref file: {os.path.relpath(ref_path, motif_dir)}")

    try:
        st_ref = gemmi.read_structure(ref_path)
        st_hit = gemmi.read_structure(hit_path)
    except Exception as e:
        print(f"  STOP — failed to parse structure: {e}")
        return

    # Stage 1: does the reference phosphate even make protein contacts?
    ref_inters_per_p = ap.get_interactions_per_phosphate(st_ref, ref_lig, ref_num)
    ref_p_positions  = ap.get_ref_p_positions(st_ref, ref_lig, ref_num)
    print(f"\n  [Stage 1] Reference phosphate groups with protein contacts: {len(ref_inters_per_p)}")
    if not ref_inters_per_p:
        print("    STOP — reference ligand has no scorable phosphate interactions.")
        print("    This reference site can never produce a PLIF result for ANY hit.")
        return
    for p_idx, inters in ref_inters_per_p.items():
        print(f"    P{p_idx}: {len(inters)} interaction(s)")
        for pos, itype, rn, rnum, ch in inters:
            print(f"       {itype:<12} {rn}{rnum}{ch}")

    # Stage 2: is the hit excluded because a genuine phosphate is already there?
    already = ap.has_phosphate_at_site(st_hit, ref_p_positions)
    print(f"\n  [Stage 2] Hit already has a genuine phosphate near the site: {already}")
    if already:
        print("    STOP — analyze_plif.py excludes this hit from mimic scoring entirely.")
        _check_recorded(ref_id, hit_id, chain, ref_lig, ref_num, motif_dir)
        return

    # Stage 3: walk every residue through the candidate-mimic filter
    print(f"\n  [Stage 3] Candidate mimic filter (analyze_plif._process_ref_site logic):")
    hit_metal_positions = ap.get_metal_positions(st_hit)
    hit_p_positions      = ap.get_phosphorus_positions(st_hit)
    passed, excluded = [], []
    for model in st_hit:
        for c in model:
            if not c.name.isupper():
                continue
            for res in c:
                if res.name in ("HOH", "DOD", "WAT"):
                    continue
                reasons = _mimic_filter_reasons(res, hit_p_positions)
                if reasons:
                    if res.name not in STANDARD_AA:
                        excluded.append((res.name, res.seqid.num, c.name, reasons))
                else:
                    passed.append((res.name, res.seqid.num, c.name))

    if not passed:
        print("    No residues passed the mimic filter.")
    for res_name, res_num, chain_name in passed:
        print(f"    PASS  {res_name}{res_num}{chain_name}")
        hit_inters = ap.get_interactions(st_hit, res_name, res_num)
        mimic_positions = [
            (a.pos.x, a.pos.y, a.pos.z)
            for model in st_hit for c in model for res in c
            if res.name == res_name and str(res.seqid.num) == str(res_num) and c.name == chain_name
            for a in res
        ]
        for p_idx, p_ref_inters in ref_inters_per_p.items():
            score, _ = ap.calculate_spatial_similarity(p_ref_inters, hit_inters, hit_metal_positions, mimic_positions)
            flag = "passes > 0.1 threshold" if score > 0.1 else "below 0.1 threshold — discarded"
            print(f"          vs P{p_idx}: score={score:.3f}  ({flag})")

    if excluded:
        print(f"\n  [Stage 3b] Non-standard residues excluded by the filter (possible false negatives):")
        for res_name, res_num, chain_name, reasons in excluded:
            print(f"    EXCLUDED  {res_name}{res_num}{chain_name}: {', '.join(reasons)}")

    # Stage 4: compare against what's actually recorded on disk
    print(f"\n  [Stage 4] Recorded pipeline outputs for this site:")
    _check_recorded(ref_id, hit_id, chain, ref_lig, ref_num, motif_dir)


def trace(ref_id, hit_id, chain, lig, num, motif_dir):
    hits_dir = os.path.join(motif_dir, "hits")
    refs_dir = os.path.join(motif_dir, "references")

    matches = _find_hit_files(hits_dir, ref_id, hit_id, chain, lig, num)
    if not matches:
        print(f"No hit files found for hit_{ref_id}_{hit_id}_* under {hits_dir}")
        return
    if len(matches) > 1 and not (chain and lig and num):
        print(f"Multiple reference sites found for {ref_id}/{hit_id} — "
              f"pass --chain/--lig/--num to inspect one, or all will be traced below.\n")
        for m in matches:
            print(f"  {os.path.relpath(m, hits_dir)}")

    for hit_path in matches:
        _trace_one(hit_path, hits_dir, refs_dir, motif_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trace why a ref/hit pair does or doesn't make it through the PLIF pipeline."
    )
    parser.add_argument("ref_id")
    parser.add_argument("hit_id")
    parser.add_argument("--chain", default=None)
    parser.add_argument("--lig", default=None)
    parser.add_argument("--num", default=None)
    parser.add_argument("--motif_dir", default=os.path.join(PROJECT_ROOT, "results", "motif_analysis"))
    args = parser.parse_args()

    trace(args.ref_id, args.hit_id, args.chain, args.lig, args.num, args.motif_dir)
