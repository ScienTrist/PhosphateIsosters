"""
Prototype: exports a DataWarrior-importable TSV (same idiom as ../../scripts/
export_to_datawarrior.py -- a leading "SMILES" column DataWarrior auto-detects
as a structure on import) but with a SECOND structure column showing just the
part of each hit/mimic ligand that actually did the isosteric mimicking: the
atoms behind whichever of its interactions matched one of the reference
phosphate group's own interactions (the same intersection that produces
prolif_plif_score in run_prolif.py).

Where the atom-level provenance comes from: run_prolif.py calls
fp.generate(..., metadata=True) for every site, which makes ProLIF attach a
metadata dict -- including {"indices": {"ligand": (idx, ...), "protein": (...)},
"distance": ..., ...} -- to every single matched interaction. run_prolif.py
saves the whole Fingerprint object (metadata included) to
prolif_fingerprint_full_manifest.pkl, and canonicalize_ifp_for_alignment()
carries that metadata through into its remapped bits -- flatten_canon_bits() is
the only place in the existing pipeline that discards it, collapsing everything
down to a bare (residue, interaction_type) presence bit for the Tanimoto/PLIF
math. This script mirrors explain_pair.py's per-pair recomputation (phosphate
group selection, residue correspondence, canonicalization) but stops one step
before flatten_canon_bits would throw the atom indices away, then uses
run_prolif.py's own _extract_submol() -- already proven in production by
phosphate_ifp.py's phosphate-group cutting -- to cut those exact atoms out of
the hit ligand as their own fragment.

The two structure columns (full ligand, isosteric fragment) are both derived
from the SAME RDKit mol/conformer that ProLIF itself scored, so they're
guaranteed self-consistent (same protonation/deprotonation state ProLIF saw,
not a separately-looked-up canonical SMILES that might depict a different
tautomer). The fragment is a plain atom-induced subgraph -- bonds crossing the
cut are just dropped, not dummy-atom-capped -- so it renders as its own small,
valence-complete molecule/ion rather than showing where it used to attach; that
was a deliberate scope cut for this prototype (see the docstring on
_extract_submol for why this is chemically safe: RDKit's sanitizer fills the
remaining valence with implicit Hs).

Usage: python export_prolif_datawarrior.py [manifest.csv] [--limit N]
                                            [--min-score S] [--out path]
"""
import argparse
import csv
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROLIF_V2_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PROLIF_V2_ROOT)
RESULTS_DIR = os.path.join(PROLIF_V2_ROOT, "results")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import run_prolif as rp  # noqa: E402
import phosphate_ifp as pif  # noqa: E402
from utils import fmt_eta  # noqa: E402
from export_to_datawarrior import CRYSTALLOGRAPHIC_SOLVENTS  # noqa: E402
from ifg import identify_functional_groups  # noqa: E402

from rdkit import Chem  # noqa: E402
import MDAnalysis as mda  # noqa: E402


def _manifest_rows(manifest_path):
    with open(manifest_path) as f:
        return {r["site_id"]: r for r in csv.DictReader(f)}


def _comparison_rows(comparison_csv):
    with open(comparison_csv, newline="") as f:
        return list(csv.DictReader(f))


def build_hit_ligand_mol(hit_row):
    """Rebuilds the hit ligand's RDKit mol EXACTLY as run_prolif.py's run_site()
    does (same selection, same safe_molecule_from_mda call) -- this is what
    guarantees atom indices coming out of the cached fingerprint's metadata
    line up with atoms in the mol returned here."""
    pocket_path = os.path.join(rp.POCKETS_DIR, f"{hit_row['site_id']}.pdb")
    u = mda.Universe(pocket_path)
    ligand = u.select_atoms(f"resid {hit_row['lig_resnum']} and chainID {hit_row['chain']} and not protein")
    if len(ligand) == 0:
        return None, f"ligand selection matched 0 atoms in {pocket_path}"
    lig_mol = rp.safe_molecule_from_mda(ligand, use_segid=False)
    return lig_mol, None


# Interaction types excluded from atom-marking entirely -- not just from which
# atoms get highlighted, but from ref_bits/hit_bits/score themselves.
# VdWContact is a pure atom-pair distance check (any element, no directionality,
# no chemical role required), so it carries far weaker evidence of genuine
# isosteric mimicry than HBAcceptor/HBDonor/Anionic/Cationic/MetalAcceptor/etc.
# Confirmed concretely, not by assumption: diagnosing hit_6RJ2_K52_402 showed
# VdWContact alone justifying a highlighted hydrogen and pulling in an entire
# unrelated sulfonamide via functional-group expansion; hit_9D6X_MTX_402 showed
# VdWContact justifying 4 of its ligand's carbons with zero chemical backing,
# out of a reference site whose own interactions were 3/4 VdWContact.
#
# Dropped from ref_bits/hit_bits together, not just from matched_bits, so the
# "recomputed score" this function returns stays internally consistent: a
# denominator that still counted ref bits that could only ever have matched via
# an excluded interaction type would understate the true fraction replicated by
# the remaining, real interaction types. This does NOT touch prolif_plif_score
# in the published CSV/dashboard (that's computed independently by run_prolif.py
# and would need its own rerun to change) -- only this atom-marking pathway's
# own recomputation, used for highlighting and its accompanying "recomputed
# score" display in the PyMOL/PNG/DataWarrior outputs.
ATOM_MARKING_EXCLUDED_INTERACTIONS = {"VdWContact"}


def _drop_excluded_interactions(bits):
    return {b for b in bits if b[1] not in ATOM_MARKING_EXCLUDED_INTERACTIONS}


def matched_hit_atom_indices(ref_row, hit_row, hit_ifp, fp, ref_group_cache, forced_p_idx=None):
    """Returns (atom_idxs: set[int], score: float|None, err: str|None).

    Mirrors explain_pair.py's _explain_prolif() (same phosphate-group selection,
    same residue correspondence, same canonicalization) but instead of reducing
    to flatten_canon_bits() and stopping there, looks up each matched bit's own
    metadata to recover which hit ligand atom(s) produced it.

    forced_p_idx: when given (and present in this reference's own groups),
    used directly instead of re-deriving "most bits matched" below -- for
    callers that already know which phosphate group produced this pair's
    score (e.g. the discover_candidates.py-driven pool builder) and need the
    highlighted atoms to always agree with it, not risk landing on a
    different group via an independent re-derivation.
    """
    mapping = rp.build_residue_correspondence(ref_row, hit_row, cutoff=rp.CA_MATCH_CUTOFF)
    if mapping is None:
        return set(), None, "no TM-align transform on file"

    canon_hit_ifp = rp.canonicalize_ifp_for_alignment(hit_ifp, protein_mapping=mapping)
    hit_bits = _drop_excluded_interactions(rp.flatten_canon_bits(canon_hit_ifp))

    ref_site_id = ref_row["site_id"]
    if ref_site_id not in ref_group_cache:
        ref_group_cache[ref_site_id] = pif.phosphate_group_ifps(ref_row, fp)
    groups, err = ref_group_cache[ref_site_id]
    if groups is None:
        return set(), None, f"phosphate_group_ifps failed: {err}"
    if not groups:
        return set(), None, "reference ligand has no phosphorus atom"

    if forced_p_idx is not None and forced_p_idx in groups:
        ref_ifp = groups[forced_p_idx]
    elif len(groups) == 1:
        ref_ifp = next(iter(groups.values()))
    else:
        ref_ifp, best_matched = None, -1
        for cand_ifp in groups.values():
            cand_bits = _drop_excluded_interactions(rp.flatten_canon_bits(rp.canonicalize_ifp_for_alignment(cand_ifp)))
            matched = len(cand_bits & hit_bits)
            if matched > best_matched:
                ref_ifp, best_matched = cand_ifp, matched

    canon_ref_ifp = rp.canonicalize_ifp_for_alignment(ref_ifp)
    ref_bits = _drop_excluded_interactions(rp.flatten_canon_bits(canon_ref_ifp))
    matched_bits = ref_bits & hit_bits
    score = len(matched_bits) / len(ref_bits) if ref_bits else None
    if not matched_bits:
        return set(), score, None

    # bit -> metadata tuple, built once per hit ifp rather than re-scanning
    # canon_hit_ifp for every matched bit.
    bit_to_metadata = {}
    for (_lig_id, prot_id), data in canon_hit_ifp.items():
        for name, metadata in data.items():
            bit_to_metadata[(str(prot_id), name)] = metadata

    atom_idxs = set()
    for bit in matched_bits:
        for meta in bit_to_metadata.get(bit, ()):
            atom_idxs.update(meta["indices"]["ligand"])
    return atom_idxs, score, None


def expand_to_functional_groups(lig_mol, atom_idxs):
    """Widens a raw ProLIF-matched atom set to include every atom in any Ertl
    functional group (ifg.py) that at least one matched atom belongs to.

    Without this, a partially-matched oxyanion looks arbitrary and confusing --
    e.g. a sulfonate where only 2 of its 3 oxygens happen to have registered a
    ProLIF interaction bit gets only those 2 highlighted, with no visual sign
    that they're part of one functional group at all. identify_functional_groups
    finds groups by graph connectivity (heteroatom clusters, not a predefined
    per-chemotype SMARTS list), so this generalizes to whatever chemotype shows
    up -- carboxylate, sulfonate, sulfate, phosphonate, tetrazole, etc. -- without
    needing one written for it in advance.

    Deliberately a separate step from matched_hit_atom_indices() rather than
    folded into it: that function's job stays "what does ProLIF's fingerprint
    say," this one's job is "widen that to whole chemistry," and callers that
    want the raw, unwidened set for some other purpose can still get it."""
    if not atom_idxs:
        return set(atom_idxs)
    fgs = identify_functional_groups(lig_mol)
    expanded = set(atom_idxs)
    for fg in fgs:
        if set(fg.atomIds) & atom_idxs:
            expanded.update(fg.atomIds)
    return expanded


def mol_to_clean_smiles(mol):
    try:
        stripped = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(stripped)
    except Exception:
        try:
            return Chem.MolToSmiles(mol)
        except Exception:
            return ""


def atom_names(mol, idxs):
    names = []
    for i in sorted(idxs):
        info = mol.GetAtomWithIdx(i).GetPDBResidueInfo()
        names.append(info.GetName().strip() if info else f"idx{i}")
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="full_manifest.csv")
    parser.add_argument("--limit", type=int, default=200,
                         help="max pairs to process, sorted by ProLIF PLIF score descending (0 = no limit)")
    parser.add_argument("--min-score", type=float, default=0.0,
                         help="skip pairs with prolif_plif_score below this")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    manifest_path = os.path.join(PROLIF_V2_ROOT, args.manifest)
    manifest_tag = os.path.splitext(os.path.basename(manifest_path))[0]
    run_suffix = "" if manifest_tag == "sample_manifest" else f"_{manifest_tag}"
    comparison_csv = os.path.join(RESULTS_DIR, f"plif_vs_tanimoto_comparison{run_suffix}.csv")
    fp_pickle = os.path.join(RESULTS_DIR, f"prolif_fingerprint{run_suffix}.pkl")
    out_path = args.out or os.path.join(RESULTS_DIR, f"prolif_isostere_datawarrior{run_suffix}.txt")

    for p in (manifest_path, comparison_csv, fp_pickle):
        if not os.path.exists(p):
            print(f"Error: {p} not found.")
            sys.exit(1)

    manifest = _manifest_rows(manifest_path)
    rows = _comparison_rows(comparison_csv)
    rows = [r for r in rows if r["prolif_plif_score"] not in ("", None)
            and float(r["prolif_plif_score"]) >= args.min_score]
    rows.sort(key=lambda r: float(r["prolif_plif_score"]), reverse=True)
    if args.limit:
        rows = rows[:args.limit]

    print(f"Loading cached ProLIF fingerprint from {fp_pickle} ...")
    import prolif as plf
    fp = plf.Fingerprint.from_pickle(fp_pickle)
    ifp_by_site = dict(zip(fp.site_ids, fp.ifp.values()))

    ref_group_cache = {}
    hit_mol_cache = {}
    out_rows = []
    n_ok = n_fail = 0
    t0 = time.time()
    total = len(rows)
    print(f"Marking isosteric atoms for {total} pairs (limit={args.limit or 'none'}, min_score={args.min_score}) ...")

    for i, row in enumerate(rows, 1):
        ref_sid, hit_sid = row["ref_site"], row["hit_site"]
        ref_row, hit_row = manifest.get(ref_sid), manifest.get(hit_sid)
        elapsed = time.time() - t0
        eta = elapsed / i * (total - i) if i else 0
        progress = f"[{i}/{total} {100*i/total:3.0f}%  ok={n_ok} fail={n_fail}]"

        if ref_row is None or hit_row is None:
            n_fail += 1
            print(f"  {progress} FAIL {ref_sid} vs {hit_sid}: not in {args.manifest} (ETA {fmt_eta(eta)})")
            continue

        hit_ifp = ifp_by_site.get(hit_sid)
        if hit_ifp is None:
            n_fail += 1
            print(f"  {progress} FAIL {ref_sid} vs {hit_sid}: {hit_sid} not in cached fingerprint (ETA {fmt_eta(eta)})")
            continue

        try:
            atom_idxs, score, err = matched_hit_atom_indices(ref_row, hit_row, hit_ifp, fp, ref_group_cache)
            if err:
                n_fail += 1
                print(f"  {progress} FAIL {ref_sid} vs {hit_sid}: {err} (ETA {fmt_eta(eta)})")
                continue

            if hit_sid not in hit_mol_cache:
                hit_mol_cache[hit_sid] = build_hit_ligand_mol(hit_row)
            lig_mol, mol_err = hit_mol_cache[hit_sid]
            if lig_mol is None:
                n_fail += 1
                print(f"  {progress} FAIL {ref_sid} vs {hit_sid}: {mol_err} (ETA {fmt_eta(eta)})")
                continue

            atom_idxs = expand_to_functional_groups(lig_mol, atom_idxs)

            full_smiles = mol_to_clean_smiles(lig_mol)
            total_heavy = sum(1 for a in lig_mol.GetAtoms() if a.GetAtomicNum() > 1)

            if atom_idxs:
                frag = rp._extract_submol(lig_mol, sorted(atom_idxs))
                frag_smiles = mol_to_clean_smiles(frag)
                names = atom_names(lig_mol, atom_idxs)
                iso_heavy = sum(1 for i2 in atom_idxs if lig_mol.GetAtomWithIdx(i2).GetAtomicNum() > 1)
            else:
                frag_smiles, names, iso_heavy = "", [], 0

            fraction = iso_heavy / total_heavy if total_heavy else 0.0

        except Exception as e:
            n_fail += 1
            print(f"  {progress} FAIL {ref_sid} vs {hit_sid}: {type(e).__name__}: {e} (ETA {fmt_eta(eta)})")
            continue

        n_ok += 1
        mimic = hit_row["lig_resname"]
        print(f"  {progress} {ref_row['pdb_id']}/{hit_row['pdb_id']} ({mimic}): "
              f"ProLIF_PLIF={score:.3f}  isosteric={iso_heavy}/{total_heavy} heavy atoms "
              f"[{','.join(names) or 'none'}] (ETA {fmt_eta(eta)})")

        out_rows.append({
            "SMILES": full_smiles,
            "Isosteric_Fragment_SMILES": frag_smiles,
            "ID": mimic,
            "Ref_PDB": ref_row["pdb_id"], "Hit_PDB": hit_row["pdb_id"],
            "Ref_Site": ref_sid, "Hit_Site": hit_sid,
            "ProLIF_PLIF_Score": round(score, 3) if score is not None else "",
            "Homebrew_PLIF_Score": round(float(row["plif_score"]), 3) if row.get("plif_score") else "",
            "Isosteric_Atom_Names": ", ".join(names),
            "Isosteric_Heavy_Atoms": iso_heavy,
            "Total_Heavy_Atoms": total_heavy,
            "Isosteric_Fraction": round(fraction, 3),
            "N_CA_Mapped": row.get("n_ca_mapped", ""),
            "Same_Phosphate_Group": row.get("same_phosphate_group", ""),
            "Crystallographic_Solvent": "Yes" if mimic in CRYSTALLOGRAPHIC_SOLVENTS else "No",
        })

    fieldnames = ["SMILES", "Isosteric_Fragment_SMILES", "ID", "Ref_PDB", "Hit_PDB", "Ref_Site", "Hit_Site",
                  "ProLIF_PLIF_Score", "Homebrew_PLIF_Score", "Isosteric_Atom_Names",
                  "Isosteric_Heavy_Atoms", "Total_Heavy_Atoms", "Isosteric_Fraction",
                  "N_CA_Mapped", "Same_Phosphate_Group", "Crystallographic_Solvent"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\n{n_ok}/{total} pairs marked, {n_fail} failed.")
    print(f"Wrote {out_path}")
    print("Open in DataWarrior: File > Open, then confirm 'SMILES' and "
          "'Isosteric_Fragment_SMILES' as chemical structure columns when prompted.")


if __name__ == "__main__":
    main()
