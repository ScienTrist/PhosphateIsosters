"""
Prototype step 2: renders each hit ligand as a 2D PNG with the isosteric atoms
(from export_prolif_datawarrior.matched_hit_atom_indices) highlighted in-place,
using RDKit's own drawing code -- sidesteps DataWarrior's highlighting
limitations entirely since the coloring happens before DataWarrior ever opens
anything.

Usage: python render_highlighted_hits.py [manifest.csv] [--limit N] [--min-score S] [--out-dir path]
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from export_prolif_datawarrior import (  # noqa: E402
    PROLIF_V2_ROOT, RESULTS_DIR, _manifest_rows, _comparison_rows,
    matched_hit_atom_indices, build_hit_ligand_mol, expand_to_functional_groups,
)

from rdkit import Chem  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402
from rdkit.Chem import rdDepictor  # noqa: E402


# Orange/magenta, not red -- red is reserved for wedge/dash stereo bonds in
# DataWarrior's own structure rendering, so a red highlight would be
# ambiguous there. Exact RGB pulled from DataWarrior's own AbstractDepictor.
# COLOR_ORANGE/COLOR_MAGENTA constants (via JPype against the local
# DataWarrior install) so these PNGs match export_ifg_dataset_datawarrior.py's
# inline SMILES_Highlight atom coloring pixel-for-pixel.
HIGHLIGHT_COLOR = (1.0, 160 / 255, 0.0)      # raw ProLIF-matched atoms -- orange
FG_HIGHLIGHT_COLOR = (192 / 255, 0.0, 1.0)   # functional-group-expansion-added atoms -- magenta
                                              # (distinguishes "backed by a real interaction" from
                                              # "completes the same functional group as one that was";
                                              # pymol_browse_isosteric_atoms.py uses its own separate
                                              # magenta/purple palette, unrelated to this one)


def render(lig_mol, atom_idxs, fg_idxs, out_path, size=(420, 340)):
    # RemoveHs renumbers atoms, so the highlight set must be translated through
    # the same old-index -> new-index map RemoveHs uses internally: atoms it
    # keeps retain their RELATIVE order, so building old_idx -> new_idx by
    # walking the original mol in order and skipping removed (numeric-name-only,
    # non-isotope) H atoms reproduces it without relying on undocumented internals.
    mol_noh = Chem.RemoveHs(lig_mol)
    kept_old_idxs = [a.GetIdx() for a in lig_mol.GetAtoms() if a.GetAtomicNum() > 1]
    old_to_new = {old: new for new, old in enumerate(kept_old_idxs)}
    mapped_idxs = {old_to_new[i] for i in atom_idxs if i in old_to_new}
    mapped_fg_idxs = {old_to_new[i] for i in fg_idxs if i in old_to_new}

    rdDepictor.Compute2DCoords(mol_noh)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    highlight_colors = {i: HIGHLIGHT_COLOR for i in mapped_idxs}
    highlight_colors.update({i: FG_HIGHLIGHT_COLOR for i in mapped_fg_idxs})
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol_noh,
        highlightAtoms=list(mapped_idxs) + list(mapped_fg_idxs),
        highlightAtomColors=highlight_colors,
    )
    drawer.FinishDrawing()
    with open(out_path, "wb") as f:
        f.write(drawer.GetDrawingText())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="full_manifest.csv")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--max-per-ligand", type=int, default=None,
                         help="cap how many pairs with the same mimic ligand code can be selected, "
                              "so --limit spreads across more distinct chemotypes instead of being "
                              "dominated by whichever ligand happens to repeat against one busy reference "
                              "(e.g. GOA hit 7 different structures of the same 6AK5 reference)")
    parser.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "highlighted_pngs"))
    args = parser.parse_args()

    manifest_path = os.path.join(PROLIF_V2_ROOT, args.manifest)
    manifest_tag = os.path.splitext(os.path.basename(manifest_path))[0]
    run_suffix = "" if manifest_tag == "sample_manifest" else f"_{manifest_tag}"
    comparison_csv = os.path.join(RESULTS_DIR, f"plif_vs_tanimoto_comparison{run_suffix}.csv")
    fp_pickle = os.path.join(RESULTS_DIR, f"prolif_fingerprint{run_suffix}.pkl")

    os.makedirs(args.out_dir, exist_ok=True)

    manifest = _manifest_rows(manifest_path)
    rows = _comparison_rows(comparison_csv)
    rows = [r for r in rows if r["prolif_plif_score"] not in ("", None)
            and float(r["prolif_plif_score"]) >= args.min_score]
    rows.sort(key=lambda r: float(r["prolif_plif_score"]), reverse=True)

    if args.max_per_ligand:
        per_ligand_count = {}
        capped = []
        for r in rows:
            lig = r["mimic"]
            if per_ligand_count.get(lig, 0) >= args.max_per_ligand:
                continue
            per_ligand_count[lig] = per_ligand_count.get(lig, 0) + 1
            capped.append(r)
        rows = capped

    if args.limit:
        rows = rows[:args.limit]

    import prolif as plf
    fp = plf.Fingerprint.from_pickle(fp_pickle)
    ifp_by_site = dict(zip(fp.site_ids, fp.ifp.values()))

    ref_group_cache = {}
    hit_mol_cache = {}
    written = []
    for row in rows:
        ref_sid, hit_sid = row["ref_site"], row["hit_site"]
        ref_row, hit_row = manifest.get(ref_sid), manifest.get(hit_sid)
        if ref_row is None or hit_row is None:
            continue
        hit_ifp = ifp_by_site.get(hit_sid)
        if hit_ifp is None:
            continue
        try:
            atom_idxs, score, err = matched_hit_atom_indices(ref_row, hit_row, hit_ifp, fp, ref_group_cache)
            if err or not atom_idxs:
                continue
            if hit_sid not in hit_mol_cache:
                hit_mol_cache[hit_sid] = build_hit_ligand_mol(hit_row)
            lig_mol, mol_err = hit_mol_cache[hit_sid]
            if lig_mol is None:
                continue
            expanded_idxs = expand_to_functional_groups(lig_mol, atom_idxs)
            fg_idxs = expanded_idxs - atom_idxs
            # Both ref_sid's and hit_sid's own resnum suffixes are needed to guarantee
            # a unique filename -- confirmed BOTH kinds of collision happen in practice:
            # the same hit ligand appearing at a different resnum in one hit structure,
            # AND the same hit matched against two different reference sites within one
            # reference structure (same ref_row['pdb_id'], different ref ligand/resnum).
            # Plain pdb_id+ligand isn't unique either way.
            ref_resnum = ref_sid.rsplit("_", 1)[-1]
            hit_resnum = hit_sid.rsplit("_", 1)[-1]
            fname = f"{ref_row['pdb_id']}_{ref_resnum}_{hit_row['pdb_id']}_{hit_row['lig_resname']}_{hit_resnum}.png"
            out_path = os.path.join(args.out_dir, fname)
            render(lig_mol, atom_idxs, fg_idxs, out_path)
            written.append((out_path, hit_row["lig_resname"], score, len(atom_idxs), len(fg_idxs)))
            print(f"  wrote {fname}  (score={score:.3f}, {len(atom_idxs)} raw + {len(fg_idxs)} fg-expanded atoms)")
        except Exception as e:
            print(f"  FAIL {ref_sid} vs {hit_sid}: {type(e).__name__}: {e}")
            continue

    print(f"\n{len(written)} PNGs written to {args.out_dir}")
    return written


if __name__ == "__main__":
    main()
