"""
Regenerates the highlighted-hit PNGs that export_ifg_dataset_datawarrior.py's
image detail column points at, straight from ifg_prolif_dataset_full_manifest.csv's
own stored red_atom_idxs/purple_atom_idxs -- no need to recompute ProLIF
matching or load the fingerprint pickle, since build_ifg_prolif_dataset.py
already did that once and the CSV is the authoritative record of the result
(see that script's docstring: "enough to regenerate a highlighted PNG for any
single row on demand later"). Reuses build_hit_ligand_mol() (same mol
construction run_prolif.py/export_prolif_datawarrior.py use, so atom indices
line up) and render_highlighted_hits.render(), so there's still only one place
that knows how to draw a highlighted ligand.

Filenames follow render_highlighted_hits.py's own convention exactly --
{ref_pdb}_{ref_resnum}_{hit_pdb}_{mimic}_{hit_resnum}.png -- so
export_ifg_dataset_datawarrior.py can compute the same filename per row and
point DataWarrior's image detail column at it.

Rows with no red_atom_idxs (no ProLIF-matched interaction at that pair -- see
ProLIF_PLIF_Score) have nothing to render and are skipped.

Runs every time export_ifg_dataset_datawarrior.py is run (see its main()) so
the PNG folder never silently drifts out of sync with, or falls short of, the
current dataset CSV -- don't rely on whatever happens to already be in
--out-dir from a previous run.

Also computes, per row, the SAME red/purple highlight but expressed as
DataWarrior's own native per-atom-color text ("orange:1,3,5; magenta:0" -- see
atom_color_cell()) instead of a PNG -- this is what lets
export_ifg_dataset_datawarrior.py's SMILES column show the highlight directly
inline in the Table View (no hovering, no detail popup; see that script's
docstring for why a PNG detail column can't do that). The PNG and the
atom-color string are two views of the identical red_atom_idxs/purple_atom_idxs
data, computed in the same pass so lig_mol is only built once per hit_site.

Usage: python generate_highlighted_pngs.py [dataset.csv] [--manifest path] [--out-dir path]
"""
import argparse
import ast
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROLIF_V2_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROLIF_V2_ROOT, "results")

sys.path.insert(0, SCRIPT_DIR)

from export_prolif_datawarrior import _manifest_rows, build_hit_ligand_mol  # noqa: E402
from render_highlighted_hits import render  # noqa: E402
from smiles_to_idcode import IdcodeConverter  # noqa: E402
from rdkit import Chem  # noqa: E402

# DataWarrior's own Molecule.cAtomColor* names (Molecule.java). Not "red" --
# DataWarrior uses red for wedge/dash stereo bonds in its own structure
# rendering, so a red atom highlight would be ambiguous with that; orange is
# the matched-atom color instead (see render_highlighted_hits.py's
# HIGHLIGHT_COLOR/FG_HIGHLIGHT_COLOR, which use these same two colors' exact
# RGB so the PNG and this inline highlight match pixel-for-pixel).
MATCHED_COLOR_NAME = "orange"
COMPLETION_COLOR_NAME = "magenta"


def png_filename(row):
    # Same collision-proofing as render_highlighted_hits.py: both sites'
    # resnum suffixes are needed, plain pdb_id+ligand isn't unique.
    ref_resnum = row["ref_site"].rsplit("_", 1)[-1]
    hit_resnum = row["hit_site"].rsplit("_", 1)[-1]
    return f"{row['ref_pdb']}_{ref_resnum}_{row['hit_pdb']}_{row['mimic']}_{hit_resnum}.png"


def atom_color_cell(lig_mol, expected_smiles, red_idxs, purple_idxs, idcode):
    """Maps red/purple atom indices (in lig_mol's own atom order, with
    explicit Hs -- same space as the CSV's red_atom_idxs/purple_atom_idxs)
    into ATOM POSITIONS IN THE IDCODE ENCODING that
    export_ifg_dataset_datawarrior.py writes for the SMILES column, e.g.
    "orange:0,2,4; magenta:1" -- this is a TWO-STEP mapping, both steps
    load-bearing:
      1. RDKit index -> position in the SMILES STRING. DataWarrior's
         OpenChemLib SmilesParser assigns atom indices in the exact order
         atoms appear in the SMILES text (verified empirically against
         RDKit's own _smilesAtomOutputOrder atom-by-atom, both by position
         and by atomic number, across single- and multi-ring examples), so
         this step is just RDKit's own bookkeeping from the SAME
         Chem.RemoveHs(lig_mol)/MolToSmiles(canonical=True) call that
         produced full_ligand_smiles in the first place.
      2. SMILES-string position -> idcode-encoded position, via
         idcode.graph_index(). THESE ARE NOT THE SAME THING -- idcode does
         NOT preserve SmilesParser's parse order (Canonizer re-ranks atoms
         canonically before encoding; verified empirically by round-tripping
         a real idcode through IDCodeParser and finding its atomic-number
         sequence differs from the parse-order one for non-trivial
         molecules). Skipping step 2 -- i.e. writing SMILES-order positions
         directly into the atomColorInfo column -- is what caused visibly
         wrong highlighting for some rows (e.g. the 4JYW/5D29 pair) even
         though the highlighted PNG (which never involves idcode at all) was
         always correct.

    Returns "" if there's nothing to highlight, or None if anything along the
    way is untrustworthy -- the recomputed SMILES doesn't match
    expected_smiles (the CSV's stored full_ligand_smiles), or idcode.graph_index
    fails/doesn't cover every atom -- so the caller should skip highlighting
    for that row rather than risk coloring the wrong atoms."""
    if not red_idxs and not purple_idxs:
        return ""
    mol_noh = Chem.RemoveHs(lig_mol)
    kept_old_idxs = [a.GetIdx() for a in lig_mol.GetAtoms() if a.GetAtomicNum() > 1]
    old_to_new = {old: new for new, old in enumerate(kept_old_idxs)}
    smiles = Chem.MolToSmiles(mol_noh, canonical=True)
    if smiles != expected_smiles:
        return None
    order = ast.literal_eval(mol_noh.GetProp("_smilesAtomOutputOrder"))
    pos_of_new_idx = {new_idx: pos for pos, new_idx in enumerate(order)}

    graph_index = idcode.graph_index(smiles)
    if len(graph_index) != mol_noh.GetNumAtoms():
        return None

    def positions(idxs):
        smiles_pos = {pos_of_new_idx[old_to_new[i]] for i in idxs
                       if i in old_to_new and old_to_new[i] in pos_of_new_idx}
        return sorted({graph_index[p] for p in smiles_pos})

    groups = []
    red_pos = positions(red_idxs)
    if red_pos:
        groups.append(f"{MATCHED_COLOR_NAME}:{','.join(str(p) for p in red_pos)}")
    purple_pos = positions(purple_idxs)
    if purple_pos:
        groups.append(f"{COMPLETION_COLOR_NAME}:{','.join(str(p) for p in purple_pos)}")
    return "; ".join(groups)


def generate(dataset_path, manifest_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(dataset_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["red_atom_idxs"]]

    manifest = _manifest_rows(manifest_path)
    idcode = IdcodeConverter()

    hit_mol_cache = {}
    atom_colors = {}
    written = failed = n_color_mismatch = 0
    total = len(rows)
    print(f"Rendering {total} highlighted PNGs (rows with a matched red fragment) -> {out_dir}")
    for i, row in enumerate(rows, 1):
        ref_sid, hit_sid = row["ref_site"], row["hit_site"]
        hit_row = manifest.get(hit_sid)
        if hit_row is None:
            print(f"  [{i}/{total}] FAIL {ref_sid} vs {hit_sid}: not in {manifest_path}")
            failed += 1
            continue
        try:
            if hit_sid not in hit_mol_cache:
                hit_mol_cache[hit_sid] = build_hit_ligand_mol(hit_row)
            lig_mol, mol_err = hit_mol_cache[hit_sid]
            if lig_mol is None:
                raise RuntimeError(mol_err)
            atom_idxs = {int(x) for x in row["red_atom_idxs"].split(",")}
            purple_idxs = {int(x) for x in row["purple_atom_idxs"].split(",")} if row["purple_atom_idxs"] else set()
            out_path = os.path.join(out_dir, png_filename(row))
            render(lig_mol, atom_idxs, purple_idxs, out_path)
            written += 1

            colors = atom_color_cell(lig_mol, row["full_ligand_smiles"], atom_idxs, purple_idxs, idcode)
            if colors is None:
                n_color_mismatch += 1
            else:
                atom_colors[(ref_sid, hit_sid)] = colors
        except Exception as e:
            print(f"  [{i}/{total}] FAIL {ref_sid} vs {hit_sid}: {type(e).__name__}: {e}")
            failed += 1
            continue
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] {written} written, {failed} failed")

    print(f"{written} PNGs written to {out_dir} ({failed} failed, {len(rows) - written - failed} skipped)")
    if n_color_mismatch:
        print(f"  {n_color_mismatch} rows had a recomputed SMILES that didn't match the CSV's "
              f"full_ligand_smiles -- inline atom-color highlighting skipped for those (PNG unaffected).")
    return out_dir, written, failed, atom_colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?",
                         default=os.path.join(RESULTS_DIR, "ifg_prolif_dataset_full_manifest.csv"))
    parser.add_argument("--manifest", default=os.path.join(PROLIF_V2_ROOT, "full_manifest.csv"))
    parser.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "highlighted_pngs_full_manifest"))
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: {args.dataset} not found.")
        sys.exit(1)
    if not os.path.exists(args.manifest):
        print(f"Error: {args.manifest} not found.")
        sys.exit(1)

    generate(args.dataset, args.manifest, args.out_dir)


if __name__ == "__main__":
    main()
