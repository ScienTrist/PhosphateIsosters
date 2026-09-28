"""
Reconstructs the sequential RCSB quality-filter funnel table for the report
(ligand match only -> +X-ray -> +Resolution<=2.7 -> +Rfree<=0.25 -> all three
combined), run fresh against the current data/components.cif and a live RCSB
query so every row comes from the same snapshot in time.

Note on the last two rows: "+Rfree<=0.25" (cumulative: ligand & xray & res &
rfree) and "All three combined" (ligand & (xray & res & rfree), i.e. exactly
query_rcsb()'s "no_po4" mode used by scripts/main.py) are the SAME boolean
filter, so they will always come out numerically identical when queried in
one sitting. In the original table they differed slightly (23,332 vs 21,791)
-- that gap was RCSB data drift between separately-run queries at different
times, not a methodology difference. This script queries everything back to
back to avoid reintroducing that drift, and reports the two rows separately
only for readability.

Usage: python analysis/quality_filter_funnel.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from ligand_detection_3D import get_phosphate_ligands_3d  # noqa: E402
from rcsbapi.search import AttributeQuery  # noqa: E402

CIF_FILE = os.path.join(PROJECT_ROOT, "data", "components.cif")
QUERY_MODE = "no_po4"
RES_LIMIT = 2.7
RFREE_LIMIT = 0.25
METHOD = "X-RAY DIFFRACTION"


def row(label, count, prev_count):
    removed = prev_count - count
    pct = 100 * removed / prev_count if prev_count else 0.0
    print(f"  {label:<28} {count:>7,}   (-{removed:,}, -{pct:.1f}%)")
    return count


def main():
    print(f"Deriving phosphate-like ligand vocabulary ({QUERY_MODE} mode) from {CIF_FILE} ...")
    phosphate_ligands = get_phosphate_ligands_3d(CIF_FILE, mode=QUERY_MODE)
    simple_phosphates = {"PO4", "PI", "2HP"}
    ligand_codes = [l for l in phosphate_ligands if l not in simple_phosphates]
    print(f"{len(ligand_codes)} candidate ligand codes\n")

    q_ligand = AttributeQuery(
        attribute="rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
        operator="in",
        value=ligand_codes,
    )
    q_xray = AttributeQuery(attribute="exptl.method", operator="exact_match", value=METHOD)
    q_res = AttributeQuery(attribute="rcsb_entry_info.resolution_combined", operator="less_or_equal", value=RES_LIMIT)
    q_rfree = AttributeQuery(attribute="refine.ls_R_factor_R_free", operator="less_or_equal", value=RFREE_LIMIT)

    print("Table 2: Sequential RCSB quality-filter funnel (fresh, single-session query)")
    print("-" * 70)

    n_ligand = len(list(q_ligand()))
    print(f"  {'Ligand match only (3D)':<28} {n_ligand:>7,}")

    n_xray = row("+ X-ray diffraction", len(list((q_ligand & q_xray)())), n_ligand)
    n_res = row("+ Resolution <= 2.7 A", len(list((q_ligand & q_xray & q_res)())), n_xray)
    n_rfree = row("+ Rfree <= 0.25", len(list((q_ligand & q_xray & q_res & q_rfree)())), n_res)

    print()
    n_combined = row("All three combined", len(list((q_ligand & (q_xray & q_res & q_rfree))())), n_ligand)

    print()
    print(f"Cumulative row (+Rfree) vs combined row match: {n_rfree == n_combined} "
          f"({n_rfree:,} vs {n_combined:,})")


if __name__ == "__main__":
    main()
