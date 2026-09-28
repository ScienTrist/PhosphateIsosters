"""
Scores each resolved (ref, hit) isostere candidate by real-PLIP interaction
fingerprint similarity.

For each pair: extract PLIP interactions for the reference ligand (own frame)
and the hit mimic ligand (own frame), bring the hit interactions into the
reference frame using the same TM-align transform the rest of the pipeline
uses, then do type-wise greedy nearest-neighbor matching (one ref interaction
matches at most one hit interaction of the same type, within a distance
cutoff) to find the intersection.

Score = Tanimoto similarity = |matched| / (|ref| + |hit| - |matched|).
This penalizes both missed reference contacts (lowers the numerator) and
extra/messy contacts the mimic makes that the reference didn't have (grows
the union) -- unlike a plain recall score, a mimic that reproduces the
reference's contacts AND adds a pile of unrelated ones doesn't score as well
as a clean, minimal match.
"""
import csv
import os
import sys
from collections import defaultdict

import gemmi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from common import OUT_DIR, find_pdb_path, get_transformation, apply_transform
from plip_extractor import extract_interactions
from phosphate_groups import get_phosphate_groups, filter_to_group
from utils import dist as _dist  # noqa: E402

MATCH_CUTOFF = 2.0  # Angstroms; anchor atoms differ slightly in choice from PLIF's old 1.5A bubble-cropped cutoff

INTERACTION_TYPES = ["HBOND", "SALTBRIDGE", "METAL", "HYDROPHOBIC",
                     "PISTACK", "PICATION", "HALOGEN", "WATERBRIDGE"]

# HBOND and SALTBRIDGE are matched against each other, not just within their
# own type: PLIP's charge perception is built around known organic functional
# groups (carboxylate, phosphate, guanidinium...), so an unusual inorganic
# oxyanion mimic (WO4, VO4, MoO4...) often isn't recognized as "charged" at
# all -- the exact same Arg contact that PLIP calls a SALTBRIDGE against the
# reference phosphate gets called a plain HBOND against the mimic. Confirmed
# concretely on 1DKP(IHP)/1DKO(WO4): ARG16/20/92 contacts sit 0.9-1.4A apart
# post-transform (clearly the same physical contact) but were being scored as
# zero matches under strict per-type matching. Per-type ref/hit/matched counts
# are still reported separately below; only the matching step merges them.
_MATCH_GROUPS = {"HBOND": "POLAR", "SALTBRIDGE": "POLAR"}


def _match_group(itype):
    return _MATCH_GROUPS.get(itype, itype)


def transform_records(records, t, u):
    out = []
    for r in records:
        r2 = dict(r)
        r2["coords"] = apply_transform(t, u, r["coords"])
        out.append(r2)
    return out


def match_fingerprints(ref_records, hit_records, cutoff=MATCH_CUTOFF):
    """Greedy nearest-neighbor 1:1 matching within each match group (see
    _MATCH_GROUPS). Returns (n_matched_total, per_type dict of
    {n_ref, n_hit, n_matched} keyed by each interaction's own original type)."""
    ref_by_group = defaultdict(list)
    hit_by_group = defaultdict(list)
    for i, r in enumerate(ref_records):
        ref_by_group[_match_group(r["type"])].append(i)
    for i, r in enumerate(hit_records):
        hit_by_group[_match_group(r["type"])].append(i)

    matched_ref_idxs, matched_hit_idxs = set(), set()
    for group in set(ref_by_group) | set(hit_by_group):
        ref_idxs = ref_by_group.get(group, [])
        hit_idxs = hit_by_group.get(group, [])
        pairs = []
        for ri in ref_idxs:
            for hi in hit_idxs:
                d = _dist(ref_records[ri]["coords"], hit_records[hi]["coords"])
                if d <= cutoff:
                    pairs.append((d, ri, hi))
        pairs.sort(key=lambda x: x[0])

        used_ref, used_hit = set(), set()
        for d, ri, hi in pairs:
            if ri in used_ref or hi in used_hit:
                continue
            used_ref.add(ri)
            used_hit.add(hi)
            matched_ref_idxs.add(ri)
            matched_hit_idxs.add(hi)

    per_type = {}
    for itype in INTERACTION_TYPES:
        ref_idxs = [i for i, r in enumerate(ref_records) if r["type"] == itype]
        hit_idxs = [i for i, r in enumerate(hit_records) if r["type"] == itype]
        per_type[itype] = {
            "n_ref": len(ref_idxs),
            "n_hit": len(hit_idxs),
            "n_matched": sum(1 for ri in ref_idxs if ri in matched_ref_idxs),
        }

    return len(matched_ref_idxs), per_type


def score_pair(row):
    ref_id, hit_id = row["Ref_ID"], row["Hit_ID"]
    ref_path = find_pdb_path(ref_id, "references")
    hit_path = find_pdb_path(hit_id, "hits")

    ref_records, err1 = extract_interactions(ref_path, row["Ref_Lig"], row["Ref_Chain"], int(row["Ref_Num"]))
    if err1:
        return None, f"ref extraction failed: {err1}"

    # Restrict the reference fingerprint to just the specific phosphate group
    # being tested against this mimic, not the whole ligand -- a multi-phosphate
    # ligand (IHP) or a phosphate-bearing cofactor (NAD/FAD/ATP) has many
    # interactions that have nothing to do with the one group a small mimic
    # could plausibly reproduce. Matches how PLIF itself is scoped.
    ref_p_idx = row.get("Ref_P_Idx", "")
    if ref_p_idx not in ("", "None", None):
        st_ref = gemmi.read_structure(ref_path)
        groups = get_phosphate_groups(st_ref, row["Ref_Lig"], int(row["Ref_Num"]))
        target_group = next((g for g in groups if str(g["p_idx"]) == str(ref_p_idx)), None)
        if target_group is not None:
            ref_records = filter_to_group(ref_records, target_group["atoms"])

    hit_records, err2 = extract_interactions(hit_path, row["Hit_Ligand"], row["Hit_Chain"], int(row["Hit_Num"]))
    if err2:
        return None, f"hit extraction failed: {err2}"

    transform = get_transformation(ref_id, hit_id)
    if transform is None:
        return None, "no TM-align transform found"
    t, u = transform
    hit_records_in_ref_frame = transform_records(hit_records, t, u)

    n_matched, per_type = match_fingerprints(ref_records, hit_records_in_ref_frame)
    n_ref, n_hit = len(ref_records), len(hit_records)
    union = n_ref + n_hit - n_matched
    tanimoto = round(n_matched / union, 4) if union > 0 else 0.0

    out_row = {
        "Ref_ID": ref_id, "Hit_ID": hit_id,
        "Ref_Lig": row["Ref_Lig"], "Ref_Num": row["Ref_Num"],
        "Hit_Ligand": row["Hit_Ligand"],
        "EC_Class": row["EC_Class"], "Metal_Status": row["Metal_Status"],
        "n_ref": n_ref, "n_hit": n_hit, "n_matched": n_matched,
        "tanimoto": tanimoto,
    }
    for itype in INTERACTION_TYPES:
        pt = per_type[itype]
        out_row[f"{itype}_ref"] = pt["n_ref"]
        out_row[f"{itype}_hit"] = pt["n_hit"]
        out_row[f"{itype}_matched"] = pt["n_matched"]
    return out_row, None


def main(limit=0):
    resolved_path = f"{OUT_DIR}/resolved_pairs.csv"
    rows = list(csv.DictReader(open(resolved_path)))
    if limit:
        rows = rows[:limit]

    out_path = f"{OUT_DIR}/plip_isostere_scores.csv"
    fieldnames = ["Ref_ID", "Hit_ID", "Ref_Lig", "Ref_Num", "Hit_Ligand", "EC_Class", "Metal_Status",
                  "n_ref", "n_hit", "n_matched", "tanimoto"]
    for itype in INTERACTION_TYPES:
        fieldnames += [f"{itype}_ref", f"{itype}_hit", f"{itype}_matched"]

    n_ok = n_skip = 0
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(rows):
            result, err = score_pair(row)
            if result is None:
                n_skip += 1
                if n_skip <= 20:
                    print(f"  [skip] {row['Ref_ID']}/{row['Hit_ID']}: {err}", file=sys.stderr)
            else:
                w.writerow(result)
                n_ok += 1
            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(rows)} processed ({n_ok} ok, {n_skip} skipped)")

    print(f"\nDone. {n_ok} scored, {n_skip} skipped. Wrote {out_path}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(limit=lim)
