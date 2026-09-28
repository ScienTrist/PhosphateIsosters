"""
Converts a build_ifg_prolif_dataset.py output CSV into a native DataWarrior
file (.dwar): SMILES and IFG_Group_SMILES render as REAL inline structure
pictures the moment the file opens -- no import wizard, no manual per-column
conversion -- because their cells hold DataWarrior's own idcode encoding (see
smiles_to_idcode.py) instead of raw SMILES text. This matches how the old
plain-.txt file behaved once you clicked through its "confirm as chemical
structure column" import wizard, minus the clicking. Red_Fragment_SMILES,
Red_Fragment_SMARTS, and IFG_Group_SMARTS all stay plain text (not asked to be
pictures).

SMILES additionally carries the orange/magenta highlight -- the SAME
red_atom_idxs/purple_atom_idxs the PNGs use -- directly on the inline
structure via a child "atom color info" column (SMILES_Highlight, see
generate_highlighted_pngs.atom_color_cell()), so the highlighting shows up
immediately in the Table View with no hovering and no detail popup.
generate_highlighted_pngs.py's PNG rendering is still run to compute this
highlighting (see atom_color_cell()), but the PNGs themselves are no longer
wired into the .dwar as a detail column -- inline highlighting on SMILES
covers that need directly in the Table View.

Why .dwar and not the older plain .txt: DataWarrior dispatches file parsing
purely by extension (see CompoundFileHelper.getFileType in DataWarrior's own
source) -- .txt always goes through the plain-text import wizard, while .dwar
is parsed natively and understands <column properties> like
columnProperty="specialType	idcode" (inline structure columns) -- confirmed
against a real DataWarrior sample file (flyingObjects.dwar) and against
DataWarrior's own CompoundTableDetailHandler.java / DWARFileParser.java /
CompoundTableConstants.java source.

Structure columns, so DataWarrior can show them inline without needing a
separately rendered PNG for browsing:
  SMILES               full mimic ligand, orange/magenta-highlighted inline
                        via its SMILES_Highlight child column (see above --
                        not red, DataWarrior reserves red for wedge/dash
                        stereo bonds in its own structure rendering)
  IFG_Group_SMILES      the whole Ertl functional group (ifg.py) those atoms
                        belong to, i.e. red + purple together, AS FOUND ON
                        THIS ROW -- blank if none; if this ligand has more
                        than one distinct group (e.g. malonate's two
                        carboxylates) the dataset CSV joins them with '; ',
                        which isn't valid SMILES on its own, so
                        smiles_to_idcode.py swaps that for '.' (disconnected-
                        component notation) before encoding -- depicts every
                        joined group side by side in one picture

Plain-text columns (deliberately NOT idcode-encoded -- see above):
  Red_Fragment_SMILES   just the atoms actually behind a matched interaction
                        (raw ProLIF match, no IFG expansion) -- blank if none
  Red_Fragment_SMARTS   same atoms as Red_Fragment_SMILES, as SMARTS -- NOT
                        canonicalized (plain per-instance Chem.MolToSmarts,
                        see summarize_ifg_groups.py's docstring for why), so
                        don't group/sort by this column expecting identical
                        fragments to match
  IFG_Group_SMARTS       a CONSISTENT SMARTS (summarize_ifg_groups.
                        smiles_to_consistent_smarts) for this row's PRIMARY
                        group -- see IFG_Group_Rank below for what "primary"
                        means and how to find the rest

Grouping structures together in DataWarrior: IFG_Group_SMARTS is the same
consistent key summarize_ifg_groups.py --key type_smarts ranks by, so two rows
with identical IFG_Group_SMARTS are guaranteed the same functional group even
when their IFG_Group_SMILES text differs (e.g. aromatic- vs alkyl-attached
carboxylate both collapse to one SMARTS -- see that script's docstring for why
this specific consolidation is correct and not an accident). Two things make
that groupable in DataWarrior without any manual setup:
  1. output ROWS are pre-sorted by IFG_Group_Rank (1 = most common group across
     the whole dataset), so identical/consistent groups already sit adjacent
     top to bottom on open -- no need to sort manually first.
  2. IFG_Group_Rank and IFG_Group_SMARTS are both plain sortable/filterable
     columns, so re-sorting after filtering, or right-clicking the column
     header for "New Row List From Category" (DataWarrior's category-grouping
     view), both work directly off them.
A ligand with more than one distinct group only gets ONE row (this script
doesn't duplicate rows), tagged with whichever of its groups is dataset-wide
more common.

Usage: python export_ifg_dataset_datawarrior.py [dataset.csv] [--out path]
                                                 [--manifest path] [--png-dir path]
                                                 [--skip-png-generation]
"""
import argparse
import csv
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROLIF_V2_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROLIF_V2_ROOT, "results")

sys.path.insert(0, SCRIPT_DIR)
from summarize_ifg_groups import build_groups, smiles_to_consistent_smarts  # noqa: E402
from generate_highlighted_pngs import generate as generate_pngs  # noqa: E402
from smiles_to_idcode import IdcodeConverter  # noqa: E402

# (DataWarrior column name, source column in the dataset CSV) -- structure
# columns first for visual proximity in the grid; IDCODE_COLUMNS below marks
# which of these render as inline pictures automatically (see module
# docstring). The derived IFG_Group_SMARTS/Rank/Instance_Count columns are
# computed separately (see main()) and spliced in right after
# IFG_Group_SMILES, not listed here.
COLUMNS = [
    ("SMILES", "full_ligand_smiles"),
    ("Red_Fragment_SMILES", "red_fragment_smiles"),
    ("Red_Fragment_SMARTS", "red_fragment_smarts"),
    ("IFG_Group_SMILES", "ifg_group_type_smiles"),
    ("Mimic", "mimic"),
    ("Ref_Ligand", "ref_ligand"),
    ("Ref_PDB", "ref_pdb"),
    ("Hit_PDB", "hit_pdb"),
    ("ProLIF_PLIF_Score", "prolif_plif_score"),
    ("Homebrew_PLIF_Score", "homebrew_plif_score"),
    ("Same_Phosphate_Group", "same_phosphate_group"),
    ("Crystallographic_Solvent", "crystallographic_solvent"),
]

INSERT_AFTER = "IFG_Group_SMILES"
DERIVED_COLUMNS = ["IFG_Group_SMARTS", "IFG_Group_Rank", "IFG_Group_Instance_Count"]

# These COLUMNS entries hold real molecules (not query patterns), so their
# cells get idcode-encoded (see smiles_to_idcode.py) instead of raw SMILES
# text, and their <column properties> mark them specialType=idcode -- that's
# what makes them render as inline pictures on open, no wizard/manual step.
IDCODE_COLUMNS = {"SMILES", "IFG_Group_SMILES"}

# Child column carrying SMILES's orange/magenta highlight (see module docstring)
# -- "parent" ties it to the SMILES column by name, DataWarrior paints those
# atoms directly onto SMILES's inline structure, no separate visible picture
# of its own.
ATOM_COLOR_COLUMN = "SMILES_Highlight"
ATOM_COLOR_PARENT = "SMILES"


def primary_group_for_row(cell, smarts_rank, smarts_count, cache):
    """Picks this row's dominant IFG group (by dataset-wide instance count)
    among however many '; '-joined groups its IFG_Group_SMILES cell holds.
    Returns (smarts, rank, instance_count) or (None, None, None) if the row
    has no group at all."""
    candidates = [s for s in cell.split("; ") if s] if cell else []
    if not candidates:
        return None, None, None
    smarts_list = [smiles_to_consistent_smarts(s, cache) for s in candidates]
    smarts_list = [s for s in smarts_list if s is not None]
    if not smarts_list:
        return None, None, None
    primary = max(smarts_list, key=lambda s: smarts_count.get(s, 0))
    return primary, smarts_rank[primary], smarts_count[primary]


def column_value(name, src, row, idcode):
    text = row.get(src, "")
    return idcode.smiles_to_idcode(text) if name in IDCODE_COLUMNS else text


def write_dwar(out_path, header, data_rows):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write("<datawarrior-fileinfo>\r\n")
        f.write('<version="3.2">\r\n')
        f.write(f'<created="{int(time.time() * 1000)}">\r\n')
        f.write(f'<rowcount="{len(data_rows)}">\r\n')
        f.write("</datawarrior-fileinfo>\r\n")
        f.write("<column properties>\r\n")
        for name in IDCODE_COLUMNS:
            f.write(f'<columnName="{name}">\r\n')
            f.write('<columnProperty="specialType\tidcode">\r\n')
        f.write(f'<columnName="{ATOM_COLOR_COLUMN}">\r\n')
        f.write('<columnProperty="specialType\tatomColorInfo">\r\n')
        f.write(f'<columnProperty="parent\t{ATOM_COLOR_PARENT}">\r\n')
        f.write("</column properties>\r\n")
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(data_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?",
                         default=os.path.join(RESULTS_DIR, "ifg_prolif_dataset_full_manifest.csv"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--manifest", default=os.path.join(PROLIF_V2_ROOT, "full_manifest.csv"))
    parser.add_argument("--png-dir", default=os.path.join(RESULTS_DIR, "highlighted_pngs_full_manifest"))
    parser.add_argument("--skip-png-generation", action="store_true",
                         help="reuse whatever is already in --png-dir instead of regenerating -- "
                              "only for quick iteration, since a stale/incomplete dir will leave "
                              "some rows without images (see generate_highlighted_pngs.py's docstring "
                              "for why this script normally regenerates every run)")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: {args.dataset} not found.")
        sys.exit(1)

    tag = os.path.splitext(os.path.basename(args.dataset))[0]
    out_path = args.out or os.path.join(RESULTS_DIR, f"{tag}_datawarrior.dwar")

    if args.skip_png_generation:
        print(f"Skipping PNG generation, reusing whatever is in {args.png_dir} "
              f"(SMILES_Highlight inline highlighting is skipped too -- it's computed "
              f"in the same pass as the PNGs)")
        atom_colors = {}
    else:
        _, _, _, atom_colors = generate_pngs(args.dataset, args.manifest, args.png_dir)

    with open(args.dataset, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Ranking IFG groups by consistent SMARTS (same logic as summarize_ifg_groups.py --key type_smarts) ...")
    groups, _, _, _ = build_groups(rows, "ifg_group_type_smiles", "ifg_group_atoms_smiles", smarts_mode=True)
    ranked = sorted(groups.items(), key=lambda kv: kv[1]["count"], reverse=True)
    smarts_rank = {smarts: rank for rank, (smarts, _) in enumerate(ranked, 1)}
    smarts_count = {smarts: g["count"] for smarts, g in groups.items()}
    print(f"  {len(ranked)} distinct groups")

    cache = {}
    enriched = []
    for row in rows:
        smarts, rank, count = primary_group_for_row(
            row["ifg_group_type_smiles"], smarts_rank, smarts_count, cache)
        enriched.append((row, smarts, rank, count))

    # Pre-sort so identical/consistent groups sit adjacent without the user
    # having to sort manually first -- ungrouped rows (rank None) go last;
    # within a group, highest recomputed score first.
    def sort_key(item):
        _, _, rank, _ = item
        score = item[0].get("prolif_plif_score") or ""
        score_val = -float(score) if score not in ("",) else 0.0
        return (rank if rank is not None else float("inf"), score_val)

    enriched.sort(key=sort_key)

    print("Encoding SMILES/IFG_Group_SMILES to DataWarrior idcode "
          "(via the local DataWarrior install's own OpenChemLib classes) ...")
    idcode = IdcodeConverter()

    out_columns = list(COLUMNS)
    insert_at = next(i for i, (name, _) in enumerate(out_columns) if name == INSERT_AFTER) + 1
    base_names = [name for name, _ in out_columns[:insert_at]]
    smiles_highlight_pos = base_names.index("SMILES") + 1
    base_names.insert(smiles_highlight_pos, ATOM_COLOR_COLUMN)
    header = base_names + DERIVED_COLUMNS + [name for name, _ in out_columns[insert_at:]]

    data_rows = []
    n_highlights = 0
    for row, smarts, rank, count in enriched:
        base_vals = [column_value(name, src, row, idcode) for name, src in out_columns[:insert_at]]
        highlight_val = atom_colors.get((row["ref_site"], row["hit_site"]), "")
        n_highlights += bool(highlight_val)
        base_vals.insert(smiles_highlight_pos, highlight_val)
        # order must match DERIVED_COLUMNS = [IFG_Group_SMARTS, IFG_Group_Rank,
        # IFG_Group_Instance_Count]
        derived_vals = [smarts or "", rank if rank is not None else "",
                         count if count is not None else ""]
        tail_vals = [column_value(name, src, row, idcode) for name, src in out_columns[insert_at:]]
        data_rows.append(base_vals + derived_vals + tail_vals)

    if idcode.n_failed:
        print(f"  {idcode.n_failed} distinct SMILES failed to encode (left blank), e.g.:")
        for smiles, err in idcode.failures:
            print(f"    {smiles!r}: {err}")

    write_dwar(out_path, header, data_rows)

    print(f"{len(rows)} rows ({n_highlights} with inline atom-color highlighting) -> {out_path}")
    print("Open in DataWarrior by double-clicking the .dwar (no import wizard, no manual per-column "
          "setup). SMILES (with orange/magenta highlighting) and IFG_Group_SMILES render inline "
          "immediately.")
    print("Rows are pre-sorted by IFG_Group_Rank, so identical functional groups are already adjacent. "
          "Right-click the IFG_Group_Rank or IFG_Group_SMARTS column header and use 'New Row List From "
          "Category' to view/color one group at a time.")


if __name__ == "__main__":
    main()
