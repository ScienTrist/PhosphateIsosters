"""
Repairs truncated residue names in extracted hit/ref CIF files.

When structures were originally extracted as PDB format, gemmi truncated residue
names longer than 3 characters (e.g. A1JEN → A1J).  This script cross-references
each extracted file against its source CIF to restore full names.

Matching strategy: residue number from the extracted file is looked up in the
source structure.  If the source has a longer name whose first 3 chars match the
extracted name, the extracted name is replaced with the full source name.
"""
import os
import sys
import glob
import gemmi
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MOTIF_DIR    = os.path.join(PROJECT_ROOT, "results", "motif_analysis")
STRUCTS_DIR  = os.path.join(PROJECT_ROOT, "data", "structures")


def find_source_structure(struct_id):
    for subdir in ["hits", "references", ""]:
        for ext in [".cif", ".pdb"]:
            p = os.path.join(STRUCTS_DIR, subdir, struct_id + ext)
            if os.path.exists(p):
                return p
    return None


def build_resnum_to_name(source_path):
    """Map (chain_auth, seqid_num) -> residue_name for all HETATM residues."""
    try:
        st = gemmi.read_structure(source_path)
    except Exception:
        return {}
    STANDARD_AA = {
        "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
        "HOH","WAT","DOD",
    }
    mapping = {}
    for model in st:
        for chain in model:
            for res in chain:
                # PDB-format sources often have entity_type=Unknown for everything;
                # use het_flag or non-standard name as the filter instead.
                is_hetatm = res.het_flag == 'H'
                is_nonstandard = res.name not in STANDARD_AA
                if is_hetatm or (is_nonstandard and res.entity_type != gemmi.EntityType.Polymer):
                    key = res.seqid.num
                    mapping[key] = res.name
    return mapping


def repair_file(extracted_path, source_path):
    try:
        st = gemmi.read_structure(extracted_path)
    except Exception as e:
        return extracted_path, f"read error: {e}", 0

    name_map = build_resnum_to_name(source_path)
    if not name_map:
        return extracted_path, "ok", 0  # source has no non-standard residues — nothing to fix

    fixes = 0
    for model in st:
        for chain in model:
            for res in chain:
                if res.entity_type not in (gemmi.EntityType.NonPolymer, gemmi.EntityType.Unknown):
                    continue
                if len(res.name) != 3:
                    continue  # already full-length or 1-2 char — skip
                source_name = name_map.get(res.seqid.num)
                if source_name and len(source_name) > 3 and source_name.startswith(res.name):
                    res.name = source_name
                    fixes += 1

    if fixes > 0:
        try:
            st.make_mmcif_document().write_file(extracted_path)
        except Exception as e:
            return extracted_path, f"write error: {e}", 0

    return extracted_path, "ok", fixes


def process_file(args):
    extracted_path, struct_id = args
    source_path = find_source_structure(struct_id)
    if not source_path:
        return extracted_path, "source not found", 0
    return repair_file(extracted_path, source_path)


def main():
    patterns = [
        os.path.join(MOTIF_DIR, "references", "ref_*.cif"),
        os.path.join(MOTIF_DIR, "hits", "**", "hit_*.cif"),
    ]

    tasks = []
    for pat in patterns:
        for path in glob.glob(pat, recursive=True):
            fname = os.path.basename(path)
            parts = fname.replace(".cif", "").split("_")
            # ref_{ref_id}_... → struct_id = parts[1]
            # hit_{ref_id}_{hit_id}_... → struct_id = parts[2] (the hit structure)
            if fname.startswith("hit_") and len(parts) >= 3:
                struct_id = parts[2].upper()
            elif fname.startswith("ref_") and len(parts) >= 2:
                struct_id = parts[1].upper()
            else:
                continue
            tasks.append((path, struct_id))

    total = len(tasks)
    print(f"Checking {total} extracted CIF files for truncated residue names...")

    ok = fixed_files = total_fixes = errors = not_found = 0
    workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_file, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            path, status, fixes = future.result()
            if status == "ok":
                ok += 1
                if fixes > 0:
                    fixed_files += 1
                    total_fixes += fixes
            elif "source not found" in status:
                not_found += 1
            else:
                errors += 1
                print(f"  ERROR: {os.path.basename(path)} — {status}")

            if i % 2000 == 0 or i == total:
                print(f"  [{i}/{total}] fixed={fixed_files} residues={total_fixes} "
                      f"errors={errors}", flush=True)

    print(f"\nDone.")
    print(f"  Files with corrected names: {fixed_files}")
    print(f"  Total residue names fixed:  {total_fixes}")
    if not_found:
        print(f"  Source structure not found: {not_found} (skipped)")
    if errors:
        print(f"  Errors: {errors}")

    if fixed_files > 0:
        print("\nRe-run analyze_plif.py and export_to_datawarrior.py to update results.")


if __name__ == "__main__":
    main()
