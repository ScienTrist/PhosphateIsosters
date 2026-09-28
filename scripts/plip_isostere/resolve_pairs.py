"""
Resolves the exact (chain, residue number) instance of each candidate mimic
ligand in the hit structure, so downstream PLIP scoring knows exactly which
ligand copy to target -- isostere_full_list.csv only records the mimic's
ligand code, not which copy of it (structures can have several).

Reuses the TM-align transform already computed for the pipeline (results/
tmalign_results_*.json) to bring hit-structure coordinates into the
reference structure's frame, then picks whichever copy of the mimic ligand
sits closest to the reference ligand -- the same logic that produced the
Distance column in isostere_full_list.csv, recomputed here as a sanity check.
"""
import csv
import os
import sys

import gemmi

import common
from common import FULL_LIST_CSV, OUT_DIR, find_pdb_path, get_transformation, apply_transform
from phosphate_groups import get_phosphate_groups

RUN_ID = "3D_no_2.7_0.25_Xray_16_0.5_0.8_7.5_0.95_0.8"


def _default_tmalign_json():
    """Prefers the from-scratch clean regeneration over the plain (older,
    pre-single-assignment-pairing) file -- never the archived _corrected.json,
    see common.py's TMALIGN_JSON comment for why."""
    results_dir = os.path.join(common.PROJECT_ROOT, "results")
    clean = os.path.join(results_dir, f"tmalign_results_{RUN_ID}_clean.json")
    plain = os.path.join(results_dir, f"tmalign_results_{RUN_ID}.json")
    if os.path.exists(clean):
        return clean
    if os.path.exists(plain):
        return plain
    raise FileNotFoundError(f"No TM-align results found for run_id={RUN_ID} (checked {clean}, {plain})")


def atom_positions(residue, exclude_ch=False):
    """All atom (x,y,z) positions in a residue. If exclude_ch, skips C/H atoms --
    matches analyze_isosteres_simple.py's candidate-ligand filter, since only
    heteroatoms (O/N/S/halogens/...) meaningfully mimic a phosphate contact."""
    return [
        (a.pos.x, a.pos.y, a.pos.z) for a in residue
        if not (exclude_ch and a.element.name in ("C", "H"))
    ]


def find_ref_ligand(structure, lig_name, lig_num):
    """Returns (chain_name, groups) for the first (chain, residue) match, where
    groups is the ligand's phosphate groups from get_phosphate_groups() (falling
    back to one pseudo-group covering all atoms if it has no P atoms) -- the
    same per-group anchor analyze_plif.py scores against, not the whole-ligand
    centroid (misleading for multi-phosphate ligands like IHP, or cofactors
    like NAD/FAD where the phosphate is a small part of a much bigger molecule).
    Logs to stderr if more than one chain has the same ligand+resnum (rare)."""
    matches = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.name == lig_name and res.seqid.num == lig_num:
                    groups = get_phosphate_groups(structure, lig_name, lig_num)
                    if not groups:
                        groups = [{"p_idx": None, "atoms": atom_positions(res)}]
                    matches.append((chain.name, groups))
        break  # first model only
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  [warn] {lig_name}:{lig_num} found in {len(matches)} chains "
              f"({[m[0] for m in matches]}); using first", file=sys.stderr)
    return matches[0]


def find_hit_ligand_candidates(structure, lig_name):
    """Returns [(chain_name, resnum, heteroatom_positions), ...] for every copy
    of lig_name, restricted to non-C/H atoms (matches the original distance metric)."""
    out = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.name == lig_name:
                    out.append((chain.name, res.seqid.num, atom_positions(res, exclude_ch=True)))
        break
    return out


def resolve_row(row):
    ref_id, hit_id = row["Ref_ID"], row["Hit_ID"]
    ref_lig, ref_num = row["Ref_Lig"], int(row["Ref_Num"])
    hit_ligand = row["Hit_Ligand"]

    ref_path = find_pdb_path(ref_id, "references", allow_cif=True)
    hit_path = find_pdb_path(hit_id, "hits", allow_cif=True)
    if ref_path is None or hit_path is None:
        return None, "missing structure file (neither .pdb nor .cif present)"

    transform = get_transformation(ref_id, hit_id)
    if transform is None:
        return None, "no TM-align transform found for this pair"
    t, u = transform

    st_ref = gemmi.read_structure(ref_path)
    ref_match = find_ref_ligand(st_ref, ref_lig, ref_num)
    if ref_match is None:
        return None, f"ref ligand {ref_lig}:{ref_num} not found in {ref_id}"
    ref_chain, ref_groups = ref_match

    st_hit = gemmi.read_structure(hit_path)
    candidates = find_hit_ligand_candidates(st_hit, hit_ligand)
    if not candidates:
        return None, f"hit ligand {hit_ligand} not found in {hit_id}"

    best = None  # (chain_name, resnum, distance, p_idx)
    for chain_name, resnum, hetero_positions in candidates:
        if not hetero_positions:
            continue
        transformed = [apply_transform(t, u, p) for p in hetero_positions]
        for group in ref_groups:
            d = min(
                sum((tp[i] - rp[i]) ** 2 for i in range(3)) ** 0.5
                for tp in transformed for rp in group["atoms"]
            )
            if best is None or d < best[2]:
                best = (chain_name, resnum, d, group["p_idx"])

    if best is None:
        return None, f"hit ligand {hit_ligand} in {hit_id} has no non-C/H atoms"
    hit_chain, hit_num, resolved_distance, ref_p_idx = best
    return {
        "Ref_ID": ref_id, "Ref_Lig": ref_lig, "Ref_Num": ref_num, "Ref_Chain": ref_chain,
        "Ref_P_Idx": ref_p_idx,
        "Hit_ID": hit_id, "Hit_Ligand": hit_ligand, "Hit_Chain": hit_chain, "Hit_Num": hit_num,
        "Resolved_Distance": round(resolved_distance, 3),
        "Stored_Distance": row["Distance"],
        "EC_Class": row["EC_Class"], "Metal_Status": row["Metal_Status"], "State": row["State"],
    }, None


def main(limit=0):
    common.set_tmalign_json(_default_tmalign_json())
    rows = [r for r in csv.DictReader(open(FULL_LIST_CSV)) if r["Hit_Ligand"] not in ("None", "", None)]
    if limit:
        rows = rows[:limit]

    out_path = f"{OUT_DIR}/resolved_pairs.csv"
    fieldnames = ["Ref_ID", "Ref_Lig", "Ref_Num", "Ref_Chain", "Ref_P_Idx", "Hit_ID", "Hit_Ligand",
                  "Hit_Chain", "Hit_Num", "Resolved_Distance", "Stored_Distance",
                  "EC_Class", "Metal_Status", "State"]

    n_ok = n_skip = 0
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(rows):
            resolved, err = resolve_row(row)
            if resolved is None:
                n_skip += 1
                if n_skip <= 20:
                    print(f"  [skip] {row['Ref_ID']}/{row['Hit_ID']} ({row['Hit_Ligand']}): {err}", file=sys.stderr)
            else:
                w.writerow(resolved)
                n_ok += 1
            if (i + 1) % 200 == 0:
                print(f"  ...{i+1}/{len(rows)} processed ({n_ok} ok, {n_skip} skipped)")

    print(f"\nDone. {n_ok} resolved, {n_skip} skipped. Wrote {out_path}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(limit=lim)
