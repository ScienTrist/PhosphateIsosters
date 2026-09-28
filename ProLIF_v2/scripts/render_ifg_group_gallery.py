"""
Visual companion to summarize_ifg_groups.py: renders one representative
example per top-ranked IFG functional group, using the SAME red/purple
highlighting convention as the earlier highlighted_pngs_novdw run
(render_highlighted_hits.render() -- red = raw ProLIF-matched atoms, purple =
IFG-completion atoms). Two outputs:

  - a single "gallery" PNG: all top-N groups in one grid, each cell titled
    with its rank/count/mimic so the whole ranked list can be scanned at a
    glance without opening dozens of files
  - one full-size PNG per group in --out-dir, for zooming into any one type

Deliberately NOT a full per-pair image dump (see build_ifg_prolif_dataset.py's
own docstring on why that was scoped out) -- this renders exactly one example
per group shown, not every pair that contributed to it.

Usage: python render_ifg_group_gallery.py [dataset.csv] [--key type|atoms]
                                           [--top N] [--out-dir path]
                                           [--gallery-out path] [--cols N]
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from export_prolif_datawarrior import (  # noqa: E402
    PROLIF_V2_ROOT, RESULTS_DIR, _manifest_rows,
    matched_hit_atom_indices, build_hit_ligand_mol, expand_to_functional_groups,
)
from render_highlighted_hits import render, HIGHLIGHT_COLOR, FG_HIGHLIGHT_COLOR  # noqa: E402
from summarize_ifg_groups import KEY_COLUMN, build_groups  # noqa: E402

import csv  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdDepictor  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402


def _infer_run_suffix(dataset_path):
    """'ifg_prolif_dataset_full_manifest.csv' -> ('_full_manifest', 'full_manifest.csv')
    'ifg_prolif_dataset.csv' (sample_manifest run) -> ('', 'sample_manifest.csv')"""
    tag = os.path.splitext(os.path.basename(dataset_path))[0]
    prefix = "ifg_prolif_dataset"
    if not tag.startswith(prefix):
        raise ValueError(f"can't infer manifest/fingerprint from unrecognized dataset filename: {dataset_path}")
    suffix = tag[len(prefix):]
    manifest_name = f"{suffix[1:]}.csv" if suffix else "sample_manifest.csv"
    return suffix, manifest_name


def _prepare_highlights(lig_mol, atom_idxs, fg_idxs):
    """Same RemoveHs-index-remap render_highlighted_hits.render() does
    internally, duplicated here (not imported) so the grid drawer can reuse
    the identical mol_noh/mapped indices without rendering each cell twice."""
    mol_noh = Chem.RemoveHs(lig_mol)
    kept_old_idxs = [a.GetIdx() for a in lig_mol.GetAtoms() if a.GetAtomicNum() > 1]
    old_to_new = {old: new for new, old in enumerate(kept_old_idxs)}
    mapped_idxs = {old_to_new[i] for i in atom_idxs if i in old_to_new}
    mapped_fg_idxs = {old_to_new[i] for i in fg_idxs if i in old_to_new}
    rdDepictor.Compute2DCoords(mol_noh)
    return mol_noh, mapped_idxs, mapped_fg_idxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?",
                         default=os.path.join(RESULTS_DIR, "ifg_prolif_dataset_full_manifest.csv"))
    parser.add_argument("--key", choices=["type", "atoms"], default="type")
    parser.add_argument("--top", type=int, default=16, help="how many top-ranked groups to render")
    parser.add_argument("--out-dir", default=None, help="directory for one full-size PNG per group")
    parser.add_argument("--gallery-out", default=None, help="path for the single combined grid PNG")
    parser.add_argument("--cols", type=int, default=4, help="grid columns for the gallery image")
    parser.add_argument("--panel-size", type=int, nargs=2, default=(320, 280), metavar=("W", "H"))
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: {args.dataset} not found.")
        sys.exit(1)

    run_suffix, manifest_name = _infer_run_suffix(args.dataset)
    manifest_path = os.path.join(PROLIF_V2_ROOT, manifest_name)
    fp_pickle = os.path.join(RESULTS_DIR, f"prolif_fingerprint{run_suffix}.pkl")
    tag = os.path.splitext(os.path.basename(args.dataset))[0]
    out_dir = args.out_dir or os.path.join(RESULTS_DIR, f"{tag}_group_examples_pngs")
    gallery_out = args.gallery_out or os.path.join(RESULTS_DIR, f"{tag}_group_gallery.png")

    for p in (manifest_path, fp_pickle):
        if not os.path.exists(p):
            print(f"Error: {p} not found.")
            sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading {args.dataset} ...")
    with open(args.dataset, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    key_field = KEY_COLUMN[args.key]
    other_field = KEY_COLUMN["atoms" if args.key == "type" else "type"]
    groups, n_rows_with_group, n_instances = build_groups(rows, key_field, other_field)
    ranked = sorted(groups.items(), key=lambda kv: kv[1]["count"], reverse=True)[:args.top]
    print(f"{len(groups)} distinct groups found; rendering top {len(ranked)}")

    manifest = _manifest_rows(manifest_path)
    print(f"Loading cached ProLIF fingerprint from {fp_pickle} ...")
    import prolif as plf
    fp = plf.Fingerprint.from_pickle(fp_pickle)
    ifp_by_site = dict(zip(fp.site_ids, fp.ifp.values()))
    ref_group_cache = {}

    grid_mols, grid_highlight_atoms, grid_highlight_colors, grid_legends = [], [], [], []
    n_ok = n_fail = 0

    for rank, (key_smiles, g) in enumerate(ranked, 1):
        pct = 100 * g["count"] / n_instances if n_instances else 0
        rendered = False
        # Try each stored example pair in order until one successfully
        # rebuilds/re-matches -- a handful of pairs can fail for unrelated
        # reasons (e.g. no TM-align transform), and falling back to the next
        # distinct example is cheap since build_groups already deduped them.
        for mimic, ref_pdb, hit_pdb, ref_sid, hit_sid in g["examples"]:
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
                lig_mol, mol_err = build_hit_ligand_mol(hit_row)
                if lig_mol is None:
                    continue
                expanded_idxs = expand_to_functional_groups(lig_mol, atom_idxs)
                fg_idxs = expanded_idxs - atom_idxs

                fname = f"{rank:02d}_{mimic}_{ref_pdb}_{hit_pdb}.png"
                out_path = os.path.join(out_dir, fname)
                render(lig_mol, atom_idxs, fg_idxs, out_path, size=tuple(args.panel_size))

                mol_noh, mapped_idxs, mapped_fg_idxs = _prepare_highlights(lig_mol, atom_idxs, fg_idxs)
                grid_mols.append(mol_noh)
                grid_highlight_atoms.append(list(mapped_idxs) + list(mapped_fg_idxs))
                colors = {i: HIGHLIGHT_COLOR for i in mapped_idxs}
                colors.update({i: FG_HIGHLIGHT_COLOR for i in mapped_fg_idxs})
                grid_highlight_colors.append(colors)
                grid_legends.append(f"#{rank} {mimic}  n={g['count']} ({pct:.0f}%)")

                print(f"  #{rank:2d} ({pct:4.1f}%, n={g['count']:3d})  {key_smiles:20s}  "
                      f"-> {fname}")
                n_ok += 1
                rendered = True
                break
            except Exception as e:
                print(f"  #{rank:2d} FAILED on {ref_sid} vs {hit_sid}: {type(e).__name__}: {e}, trying next example")
                continue
        if not rendered:
            n_fail += 1
            print(f"  #{rank:2d} {key_smiles}: FAILED -- no example pair could be rendered")

    print(f"\n{n_ok}/{len(ranked)} groups rendered, {n_fail} failed.")
    print(f"Individual PNGs -> {out_dir}")

    if grid_mols:
        w, h = args.panel_size
        n_rows = (len(grid_mols) + args.cols - 1) // args.cols
        drawer = rdMolDraw2D.MolDraw2DCairo(w * args.cols, h * n_rows, w, h)
        opts = drawer.drawOptions()
        opts.legendFontSize = 18
        drawer.DrawMolecules(
            grid_mols,
            highlightAtoms=grid_highlight_atoms,
            highlightAtomColors=grid_highlight_colors,
            legends=grid_legends,
        )
        drawer.FinishDrawing()
        with open(gallery_out, "wb") as f:
            f.write(drawer.GetDrawingText())
        print(f"Gallery grid -> {gallery_out}")


if __name__ == "__main__":
    main()
