"""
Generates PLIP's own native report (txt + xml) for one example reference
structure -- an intermediate-step artifact showing what real-PLIP output
looks like *before* plip_extractor.py canonicalizes it into the pipeline's
flat interaction records.

Uses 10GW/6PG:A:401 -- the same reference ligand already exercised in
results/motif_analysis/plif_results.json -- so the report's per-interaction
numbers can be cross-checked against the pipeline's existing PLIF results
for that structure.
"""
import os

from plip.exchange.report import StructureReport
from plip.structure.preparation import create_folder_if_not_exists

from common import OUT_DIR, find_pdb_path
from plip_extractor import load_complex

EXAMPLE_PDB_ID = "10GW"
REPORT_DIR = os.path.join(OUT_DIR, "example_report")


def main():
    pdb_path = find_pdb_path(EXAMPLE_PDB_ID, "references")
    if pdb_path is None:
        raise SystemExit(f"{EXAMPLE_PDB_ID}: no .pdb file found under data/structures/references")

    create_folder_if_not_exists(REPORT_DIR)

    mol = load_complex(pdb_path)  # PDBComplex, already .analyze()'d
    mol.output_path = REPORT_DIR + os.sep

    report = StructureReport(mol, outputprefix=f"{EXAMPLE_PDB_ID}_report")
    report.write_txt()
    report.write_xml()

    print(f"Wrote {REPORT_DIR}{os.sep}{EXAMPLE_PDB_ID}_report.txt")
    print(f"Wrote {REPORT_DIR}{os.sep}{EXAMPLE_PDB_ID}_report.xml")


if __name__ == "__main__":
    main()
